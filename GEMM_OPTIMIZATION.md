# GEMM 最適化方針（メモリアクセス中心・WMMAなし）

## 結論サマリ
- 現状の32×32タイル/1出力/2段同期の共有メモリGEMMは、レイテンシ隠蔽が弱く、共有メモリのバンクコンフリクトとスレッド過多(1024/blk)でSM資源が詰まり、約0.81 TFLOPSに頭打ち。
- まずは「メモリアクセスのみ」で効く4点を優先: 共有メモリのパディング/転置、レジスタタイル化(出力複数要素/スレッド)、ベクトルロードの完全化、ダブルバッファ＋非同期コピー(cp.async)でロード・計算をオーバーラップ。
- これらで2.5〜5倍（目安: ~60–70 ms、マシン/実装次第で更に短縮）を狙える。WMMA/TF32は最後の手段として温存。

## 現状のボトルネック
- 共有メモリアクセス: `As[ty][j]` と `Bs[j][tx]` はワープ内で列方向アクセスが多く、32バンクで衝突が起きやすい。
- スレッド構成: `TILE_SIZE=32` で `blockDim=32×32=1024`。1スレッド=1出力のためレジスタ再利用が乏しく、同期は各Kタイルごとに2回。占有率は1024thr/blk起因のレジスタ圧で頭打ちになりやすい。
- ベクトルロード: `float4`カーネルは`tx%4==0`だけがロード担当で分岐・整列判定が多い割に、計算側の畳み込みが単発(1要素/スレッド)なのでロード隠蔽に効きにくい。
- パイプライニング無し: グローバル→共有ロードとFMAが直列。SM 8.9(Ada)の強みである`cp.async`(L2→SMEM)が未使用。

## 優先度付き最適化メニュー（WMMA無し）
1) 共有メモリのバンク衝突回避  
- 方法: 共有メモリに+1または+8のスキューを入れるか、`As`を共有メモリに転置格納（loadは`[ty][tx]`、computeは`[j][ty]`参照を`[ty][j]`参照に置換）。  
- 例: `__shared__ float As[BM][BK + 8]; __shared__ float Bs[BK][BN + 8];`（8は128B境界を意識した実戦値）

2) スレッド/タイルの再設計（レジスタタイル化）  
- 目標: 1スレッドがCの`RM×RN`要素を保持（例: `RM=8, RN=4`）し、レジスタで再利用。  
- 推奨ブロック: `BM×BN=128×128` か `128×64`、`BK=8 or 16`、スレッド数256（=8 warps）。  
- 利点: 同じ共有タイルから複数出力を一気に更新→FMA/ロード比上昇、同期回数減。

3) ベクトルロードの徹底と整列  
- 共有タイルへのロードを全スレッドで均等分担し、`int4/float4`で16Bロードを連続128B整列に揃える。  
- ポインタ整列: `cudaMalloc`は十分に整列されるので、行幅とオフセットを16B境界に合わせ、端は分岐なしのゼロフィル/ガードで処理。  
- 共有配列は`__align__(16)`で宣言。

4) ダブルバッファ＋非同期コピー（`cp.async` / `cuda::memcpy_async`）  
- 2面の共有タイル `stage=0/1` を用意し、計算中に「次タイル」をL2→SMEMへ事前投入。  
- Ampere/Adaでは`cp.async`＋`commit_group/wait_group`でロードとFMAをほぼオーバーラップ可能。  
- 期待効果: レイテンシ隠蔽が大きく、単独で1.5〜2倍に到達するケースが多い。

5) 同期縮小とwarp単位最適化  
- 同期はタイルごとに1回へ（ダブルバッファ+パイプラインで可能）。  
- 片方のオペランド（AまたはB）をwarp内ブロードキャスト（`__shfl_sync`）に寄せ、共有メモリアクセスを半減する設計も有効。

6) コンパイル/実行時チューニング  
- `#pragma unroll` をK内ループ（`BK=8/16`）に付与。  
- `__restrict__`は既に付与済みだが、`__launch_bounds__(threads_per_block, min_blocks_per_sm)`でレジスタ配分をコントロールし、占有率を調整。  
- ブロック形状（`BM/BN/BK, RM/RN`）の自動探索（少数候補）で最適点を探索。

## 実装ガイド（スケルトン）

### 共有バッファ（パディング＋2面）
```cpp
__shared__ __align__(16) float As[2][BM][BK + SKEW];
__shared__ __align__(16) float Bs[2][BK][BN + SKEW];
```

### スレッド配置
- 例: `blockDim = dim3(32, 8)`(=256)でwarpあたり縦横に割り付け、各スレッドが`RM×RN`のCを保持。

### cp.async（概念）
- 先頭で`stage=0`を`cp.async`でプレロード → `commit_group` → `wait_group 0` → `__syncthreads()`  
- ループ内: 計算（`stage s`）と同時に（`stage s^1`）へ次タイルを`cp.async`、最後に`commit`し、計算完了直前で`wait_group` → `__syncthreads()`で切替。

### バンク衝突回避
- Compute時の参照は`As[s][row][k]`/`Bs[s][k][col]`で、二次元目に`+SKEW`を確保。  
- 転置格納を行う場合はload時に`As[s][k][row]`へ置く。

### 短い疑似コード
- 事前: 各スレッドが`Creg[RM][RN]=0`を確保  
- for `tile_k`:
  - 非同期: 次タイルのA/Bを`cp.async`で`As[s^1]`/`Bs[s^1]`へ（端はゼロフィル）
  - 同期済み`As[s], Bs[s]`で`k=0..BK-1`を`#pragma unroll`してFMA（`Creg`更新）
  - `commit_group` → `wait_group 0` → `__syncthreads()` → `s ^= 1`
- 終了: `Creg`をグローバルCにストア（ベクトル化可能）

## 計測で見る指標（Nsight Compute）
- 共有メモリ: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_[ld/st]`
- メモリ効率: `gld/trans_throughput`, `gld_efficiency`, `dram__bytes_read.sum.per_second`
- スケジューラ: `sm__warps_active.avg.pct_of_peak_sustained_active`, `smsp__sass_average_branch_targets_threads_uniform.pct`
- パイプライン: `smsp__inst_executed_pipe_fma/ldst`, `sm__pipe_fma_cycles_active`

## 想定効果（4090 Laptop, cc 8.9）
- 共有メモリ衝突解消＋レジスタタイル化のみ: 1.5〜2.5倍（170 ms → 70–110 ms）
- `cp.async`ダブルバッファ追加: さらに1.3〜2倍（→ 40–80 msレンジ）
- cuBLAS(約20.7 ms)には未到達でも、メモリアクセス起点の最適化だけで大幅短縮は可能。

## すぐ着手できる順序
- ステップ1: 共有メモリに`+SKEW`パディングを入れ、`As`を共有メモリ転置格納に変更（同期/ループは現状維持）。
- ステップ2: タイルを`BM×BN=128×64, BK=8`、`blockDim=256`に変更し、各スレッド`RM×RN=8×4`のレジスタタイルに。
- ステップ3: `float4`ベクトルロードをブロック全体で均等分担する形に書き換え（端はゼロフィル）。
- ステップ4: `cp.async`ダブルバッファを導入（SM 8.x専用パス、フォールバックは同期ロード）。

