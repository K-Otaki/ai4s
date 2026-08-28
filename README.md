# AI for Science のためのコンテナ講習会演習

PBS ジョブスケジューラと Apptainer を利用し、Python、PyTorch、分散 GPU 計算、ローカル LLM を実行するための演習用リポジトリです。

## ディレクトリ構成

```text
.
├── Exercise1/
│   ├── job.sh              # Python を直接／Apptainer 内で実行する PBS ジョブ
│   ├── mycontainer.def     # Python コンテナの Apptainer 定義
│   └── sample.py           # モンテカルロ法による円周率の推定
└── Exercise2/
    ├── build_pytorch.sh    # NVIDIA PyTorch コンテナの取得
    ├── build_vllm.sh       # vLLM コンテナの取得
    ├── pytorch-opencode.def # PyTorch + OpenCode コンテナの定義
    ├── mnist.py            # CNN による MNIST 分類
    ├── mnist.sh            # MNIST（単一 GPU）
    ├── mnist_multigpu.sh   # MNIST（2 ノード、DDP）
    ├── heat2d.py           # 2 次元熱方程式を学習する FNO
    ├── heat2d.sh           # FNO（2 ノード、DDP）
    ├── llm.sh              # vLLM と OpenCode の連携実行
    └── workspace/
        └── prompt.txt      # OpenCode に渡す実装課題
```

## 前提環境

- PBS Professional 互換のジョブスケジューラ（`qsub`、`mpirun`）
- Apptainer の environment module
- GPU 計算ノードと NVIDIA ドライバ
- コンテナイメージや MNIST データを取得するためのネットワーク接続

各ジョブスクリプトは、講義環境のキュー `lecture-g` とグループ `gt00` を指定しています。別の環境で使用する場合は、冒頭の `#PBS` オプションを変更してください。

ジョブは、原則として対象の演習ディレクトリから投入します。

## Exercise 1: Python と Apptainer

### Python で直接実行

```bash
cd Exercise1
module load python
python3 sample.py 1000000
```

引数はモンテカルロ法のサンプル数です。省略時は `100000000` が使用されます。

### コンテナを利用して実行

Docker Hub の Python イメージをそのまま取得する場合:

```bash
cd Exercise1
module load apptainer
apptainer pull python.sif docker://python:3.12
apptainer exec python.sif python3 sample.py 1000000
```

定義ファイルから独自イメージを作成する場合:

```bash
apptainer build mycontainer.sif mycontainer.def
apptainer run mycontainer.sif 1000000
```

PBS ジョブは次のように投入します。

```bash
qsub job.sh
```

## Exercise 2: PyTorch と分散 GPU 計算

以降のコマンドは `Exercise2` で実行します。

```bash
cd Exercise2
```

### PyTorch コンテナの準備

取得処理自体を PBS ジョブとして実行します。

```bash
qsub build_pytorch.sh
```

正常に完了すると `pytorch.sif` が作成されます。ビルド用のキャッシュと一時ファイルは `apptainer-cache/`、`apptainer-tmp/` に置かれます。

### MNIST 分類

単一 GPU:

```bash
qsub mnist.sh
```

2 ノード × 1 GPU の DistributedDataParallel（DDP）:

```bash
qsub mnist_multigpu.sh
```

初回実行時は MNIST データセットを `Exercise2/data/` に取得します。学習後の主な生成物は次のとおりです。

```text
fig/mnist_training_curve.png
fig/mnist_epoch_time.png
fig/mnist_predictions.png
models/mnist.pt
```

### 2 次元熱方程式（FNO）

`heat2d.py` は、有限差分法で作成したデータを使って Fourier Neural Operator（FNO）を学習します。標準のジョブ設定は 2 ノード × 1 GPU です。

```bash
qsub heat2d.sh
```

主な生成物:

```text
models/heat2d_fno.pt
fig/heat2d_fno_result.png
```

単一 GPU で試す場合は、`heat2d.sh` 内の単一 GPU 用コマンドを有効にし、PBS リソース指定も `select=1` に変更してください。

## ローカル LLM と OpenCode

この演習では、vLLM で Qwen3 Coder を OpenAI 互換 API として起動し、同じジョブ内の OpenCode から利用します。

### 1. vLLM コンテナの準備

```bash
qsub build_vllm.sh
```

完了すると `vllm.sif` が作成されます。

### 2. OpenCode コンテナの準備

```bash
module load apptainer
apptainer build pytorch-opencode.sif pytorch-opencode.def
```

ベースイメージを取得できない場合は、`pytorch-opencode.def` の `From:` を利用可能な NVIDIA PyTorch イメージ名に合わせてください。

### 3. モデルの配置と実行

`llm.sh` はモデルを次の場所から読み込みます。

```text
Exercise2/models/Qwen3.8-27B/
```

モデル、`vllm.sif`、`pytorch-opencode.sif` を準備した後にジョブを投入します。

```bash
qsub llm.sh
```

処理の流れは次のとおりです。

1. GPU 上で vLLM サーバーを `127.0.0.1:8000` に起動する
2. API の準備完了を待つ
3. `workspace/prompt.txt` を OpenCode に渡す
4. OpenCode が `workspace/` 内で CUDA + MPI のサンプルを実装・検証する
5. vLLM サーバーを終了する

ログは `workspace/vLLM.log` と `workspace/OpenCode.log` に保存されます。OpenCode が作成する `workspace/CMakeLists.txt` と `workspace/multi_gpu_sum.cu` は生成物として Git の管理対象外です。

## 注意事項

- `*.sif`、モデル重み、キャッシュ、ログ、学習結果のモデルは `.gitignore` の対象です。
- DDP ジョブでは `MASTER_PORT=29500` を使用します。同じノードで競合する場合は変更してください。
- `mnist.py` は CUDA GPU を前提としています。
- 実行時間や必要メモリは、GPU、ノード数、モデルサイズによって変わります。
