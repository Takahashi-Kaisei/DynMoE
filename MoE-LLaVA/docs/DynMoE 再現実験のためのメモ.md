# DynMoE 再現実験のためのメモ

## 目的

このメモは、`DynMoE/MoE-LLaVA` 系の再現実験を進めるときに必要だった手順、設定変更、ハマりどころ、確認観点を整理したものです。時系列メモではなく、後から見返してそのまま作業に入れる形にまとめています。

## 全体像

### モデルの系譜

- 祖先となるベースモデルは `LLaVA-1.5`
- MoE 型のベースモデルは `MoE-LLaVA`
- `MoE-LLaVA` の言語モデル候補
  - `StableLM`
  - `Qwen`
  - `Phi`
- `MoE-LLaVA` では、LLM の MLP 部分を MoE に置き換えて full fine-tuning する
- `DynMoE` のベースも基本的にはこの系譜に乗っている
  - 実験上は VLM 以外もある
  - 主な変更点は DeepSpeed 周辺の改良

### 学習ステージ

DynMoE のトレーニングは 3 stage 構成。
ソース: `DynMoE/MoE-LLaVA/docs/TRAIN.md`

1. `stage1`: pre-training
2. `stage2`: tuning
3. `stage3`: MoE-tuning

DynMoE では、`stage2` まで終わった学習済み LLaVA を持ってきて、そこから `stage3` の MoE-tuning を行う。

## 事前に揃えるもの

### Stage2 チェックポイント

Hugging Face の `LanguageBind/MoE-LLaVA-StableLM-Stage2` を取得して、`checkpoints/` 配下に置く。

```bash
hf download LanguageBind/MoE-LLaVA-StableLM-Stage2 \
  --local-dir ./checkpoints/MoE-LLaVA-StableLM-Stage2
```

参照先:
<https://huggingface.co/LanguageBind/MoE-LLaVA-StableLM-Stage2/tree/main>

### CLIP 重み

`MoE-LLaVA` 配下で CLIP も取得しておく。

```bash
hf download openai/clip-vit-large-patch14-336 \
  --local-dir ./openai/clip-vit-large-patch14-336
```

### データセット

- データセット: <https://huggingface.co/datasets/LanguageBind/MoE-LLaVA>
- `git clone` でも取れるが、 `hf download` やらない意味がない。
- datasetの場合は `--repo-type dataset` を付ける必要がある点に注意。
	- ここらへんは--repo typeが受け取るものを見ておく。
- --local-dir は保存先を指定する。これをやらないとキャッシュの中に入ってしまう。

```bash
hf download LanguageBind/MoE-LLaVA --repo-type dataset --local-dir /raw/
```

## 環境構築メモ

### 基本方針

- Dockerfile は最小限でよい。後でuvで入れまくってなんとかなる。
- `cudnn` だけ要件を満たすようにして、残りは `uv` に任せる運用が楽
- README に書かれている `pip` 系コマンドは、基本的に `uv` を付ければ流用できる
- `postCreateCommand` に詰め込むより、セットアップは手動実行の方が安定しやすいと思われる。

### Hugging Face CLI

`hf` コマンドを使うために先にインストールする。

```bash
uv tool install huggingface_hub
```

### 補足

- コンテナ内の `data/` にデータセットを直接ダウンロードしてしまうのが楽。
- サーバー側の raid ディレクトリを devcontainer の mount で見せる運用にすると良き。今のdevcontainer はそうなってる。

## セットアップ手順

コンテナのセットアップ後、`DynMoE/MoE-LLaVA` に移動してセットアップスクリプトを実行する。
これによって必要な Python パッケージのインストールがすむ。

```bash
cd DynMoE/MoE-LLaVA
bash setup_moe_llava.sh
```

## データセット準備

### 想定ディレクトリ構成

```text
IMAGE_FOLDER
├── llava_image
├── llava_image_tune
├── lvis_tune
├── lrv_tune
├── svit_tune
└── mimicit_tune
    └── LA
```

### finetune_dynmoe.shでどの image データを使うか

対象にする image データは `llava_image_tune` でよさそう。
元データの案内元として、以下のリンクをメモしておく。
これがDynMoEのfinetuneで使われているらしく、リンクを開いてみると `llava_image_tune` という名前のファイルが見える。

<https://pan.baidu.com/s/1xC9E6VuOOEBV5iieve0Z7A?pwd=2o0a>

ただし、学習スクリプト側では `llava_image_tune` にパスが通っているため、実際には `DATASET_DIR="${APP}/data/raw/MoE-LLaVA-unzipped"` を渡せば問題なさそう。

### 簡易セットアップ用スクリプト案

元メモに残っていた簡易版。流れの確認用。

```bash
set -Eeuo pipefail

hf download LanguageBind/MoE-LLaVA --repo-type dataset --local-dir /raw/

mkdir MoE-LLaVA-unzipped
cd MoE-LLaVA

cp -r llava_image.zip llava_image_tune_2.zip.001 llava_image_tune_2.zip.002 lvis_tune.zip lrv_tune.zip svit_tune.zip mimicit_tune train_json ../MoE-LLaVA-unzipped

cd ../MoE-LLaVA-unzipped

unzip llava_image.zip
rm llava_image.zip

cat llava_image_tune_2.zip.001 llava_image_tune_2.zip.002 > llava_image_tune_2.zip
unzip llava_image_tune_2.zip
unzip llava_image_tune.zip
rm llava_image_tune_2.zip.001 llava_image_tune_2.zip.002 llava_image_tune_2.zip llava_image_tune.zip

unzip lvis_tune.zip
rm lvis_tune.zip

unzip lrv_tune.zip
rm lrv_tune.zip

unzip svit_tune.zip
rm svit_tune.zip

unzip mimicit_tune/LA.zip -d /mimicit_tune
rm mimicit_tune/LA.zip
```

### 未実行だが整理された版

元メモに残っていた、ログ付きの `prepare_moellava_dataset.sh` 案。まだ実行はしていないが、再利用しやすい形になっているので残す。

```bash
#!/usr/bin/env bash

set -Eeuo pipefail

RAW_DIR="${1:-/raw}"
DATASET_ID="${2:-LanguageBind/MoE-LLaVA}"
DOWNLOAD_DIR="${RAW_DIR}/MoE-LLaVA"
WORK_DIR="${RAW_DIR}/MoE-LLaVA-unzipped"
LOG_DIR="${RAW_DIR}/logs"
LOG_FILE="${LOG_DIR}/prepare_moellava_dataset_$(date +%Y%m%d_%H%M%S).log"

TOTAL_STEPS=8
CURRENT_STEP=0
SCRIPT_START_TIME="$(date +%s)"

mkdir -p "${LOG_DIR}"

log() {
  local level="$1"
  shift
  local message="$*"
  printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${level}" "${message}" | tee -a "${LOG_FILE}" >&2
}

progress_bar() {
  local current="$1"
  local total="$2"
  local width=30
  local filled=$(( current * width / total ))
  local empty=$(( width - filled ))
  local bar

  bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')"
  bar="${bar}$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  printf '[%s] %d/%d\n' "${bar}" "${current}" "${total}" | tee -a "${LOG_FILE}" >&2
}

step_start() {
  CURRENT_STEP=$(( CURRENT_STEP + 1 ))
  STEP_START_TIME="$(date +%s)"
  log INFO "Step ${CURRENT_STEP}/${TOTAL_STEPS}: $*"
  progress_bar "${CURRENT_STEP}" "${TOTAL_STEPS}"
}

step_done() {
  local step_elapsed
  step_elapsed=$(( "$(date +%s)" - STEP_START_TIME ))
  log INFO "Completed in ${step_elapsed}s"
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    log ERROR "Required command not found: ${command_name}"
    exit 1
  fi
}

run_quiet() {
  local description="$1"
  shift
  log INFO "${description}"
  "$@" >>"${LOG_FILE}" 2>&1
}

cleanup_on_error() {
  local exit_code="$1"
  local line_no="$2"
  log ERROR "Stopped at line ${line_no} with exit code ${exit_code}. See ${LOG_FILE}"
  exit "${exit_code}"
}

trap 'cleanup_on_error $? ${LINENO}' ERR

extract_zip() {
  local archive_path="$1"
  local dest_dir="$2"
  run_quiet "Extracting $(basename "${archive_path}") -> ${dest_dir}" unzip -q "${archive_path}" -d "${dest_dir}"
}

require_command hf
require_command unzip
require_command cp
require_command cat
require_command rm

log INFO "Log file: ${LOG_FILE}"
log INFO "RAW_DIR=${RAW_DIR}"
log INFO "DOWNLOAD_DIR=${DOWNLOAD_DIR}"
log INFO "WORK_DIR=${WORK_DIR}"

step_start "Download dataset from Hugging Face"
run_quiet "Downloading ${DATASET_ID} into ${DOWNLOAD_DIR}" \
  hf download "${DATASET_ID}" --repo-type dataset --local-dir "${DOWNLOAD_DIR}"
step_done

step_start "Create working directory"
run_quiet "Recreate ${WORK_DIR}" rm -rf "${WORK_DIR}"
run_quiet "Create ${WORK_DIR}" mkdir -p "${WORK_DIR}"
step_done

step_start "Copy required archives"
run_quiet "Copy archives into ${WORK_DIR}" \
  cp -r \
  "${DOWNLOAD_DIR}/llava_image.zip" \
  "${DOWNLOAD_DIR}/llava_image_tune_2.zip.001" \
  "${DOWNLOAD_DIR}/llava_image_tune_2.zip.002" \
  "${DOWNLOAD_DIR}/lvis_tune.zip" \
  "${DOWNLOAD_DIR}/lrv_tune.zip" \
  "${DOWNLOAD_DIR}/svit_tune.zip" \
  "${DOWNLOAD_DIR}/mimicit_tune" \
  "${WORK_DIR}/"
step_done

step_start "Extract llava_image.zip"
extract_zip "${WORK_DIR}/llava_image.zip" "${WORK_DIR}"
run_quiet "Remove ${WORK_DIR}/llava_image.zip" rm -f "${WORK_DIR}/llava_image.zip"
step_done

step_start "Rebuild and extract llava_image_tune"
run_quiet "Combine split archive into ${WORK_DIR}/llava_image_tune_2.zip" \
  bash -lc "cat '${WORK_DIR}/llava_image_tune_2.zip.001' '${WORK_DIR}/llava_image_tune_2.zip.002' > '${WORK_DIR}/llava_image_tune_2.zip'"
extract_zip "${WORK_DIR}/llava_image_tune_2.zip" "${WORK_DIR}"
extract_zip "${WORK_DIR}/llava_image_tune.zip" "${WORK_DIR}"
run_quiet "Remove intermediate llava_image_tune archives" \
  rm -f \
  "${WORK_DIR}/llava_image_tune_2.zip.001" \
  "${WORK_DIR}/llava_image_tune_2.zip.002" \
  "${WORK_DIR}/llava_image_tune_2.zip" \
  "${WORK_DIR}/llava_image_tune.zip"
step_done

step_start "Extract lvis_tune.zip"
extract_zip "${WORK_DIR}/lvis_tune.zip" "${WORK_DIR}"
run_quiet "Remove ${WORK_DIR}/lvis_tune.zip" rm -f "${WORK_DIR}/lvis_tune.zip"
step_done

step_start "Extract lrv_tune.zip and svit_tune.zip"
extract_zip "${WORK_DIR}/lrv_tune.zip" "${WORK_DIR}"
run_quiet "Remove ${WORK_DIR}/lrv_tune.zip" rm -f "${WORK_DIR}/lrv_tune.zip"
extract_zip "${WORK_DIR}/svit_tune.zip" "${WORK_DIR}"
run_quiet "Remove ${WORK_DIR}/svit_tune.zip" rm -f "${WORK_DIR}/svit_tune.zip"
step_done

step_start "Extract mimicit_tune/LA.zip"
run_quiet "Create ${WORK_DIR}/mimicit_tune" mkdir -p "${WORK_DIR}/mimicit_tune"
extract_zip "${WORK_DIR}/mimicit_tune/LA.zip" "${WORK_DIR}/mimicit_tune"
run_quiet "Remove ${WORK_DIR}/mimicit_tune/LA.zip" rm -f "${WORK_DIR}/mimicit_tune/LA.zip"
step_done

total_elapsed=$(( "$(date +%s)" - SCRIPT_START_TIME ))
log INFO "All steps completed in ${total_elapsed}s"
log INFO "Prepared dataset directory: ${WORK_DIR}"
```

## 学習前に見直す設定

### DeepSpeed 設定

`zero2_offload.json` は optimizer を CPU 側に offload する設定なので、DeepSpeed が `DeepSpeedCPUAdam` を使おうとしてエラーになることがある。

A100 環境なら offload は不要なので、以下の差し替えがよい。

```bash
--deepspeed ./scripts/zero2_offload.json
```

を

```bash
--deepspeed ./scripts/zero2.json
```

に変更するのもあり。

しかし、忠実に再現をしたいなら、`zero2_offload.json`を使ったほうが良いかも。
上記のエラーはflash-attentionのインストールの過程で治ったし。

### `model_max_length`

サンプルの token 数が 439 だったため、smoke test 用でも `512` 以上にしておいた方が安全。
バッファを見て `2048` にしておくのもあり。

```bash
--model_max_length 512
```

## ハマりどころ

### `setuptools` 依存エラー

普通に smoke test すると、`setuptools` 依存関係が解決されずエラーになることがあった。
その場合はダウングレードで通ることがある。

```bash
uv pip install "setuptools<80"
```

また、後段の `flash-attn` 導入では次の指定で通った。よってこれでよし。

```bash
uv pip install "setuptools<82"
```

### shared memory 不足

shared memory の割り当て不足エラーが出た。
これは `docker run` 側の設定なので、devcontainer の `runArgs` に `--shm-size` を追加する。

```json
"runArgs": [
  "--shm-size",
  "64g"
]
```

意味としては、shared memory に 64GB 割り当てる設定。

## 学習時間短縮

`flash-attn` を使える環境に持っていくと学習時間短縮に効く。
最終的に通った手順は以下。

```bash
uv pip install "setuptools<82"
uv pip install "flash-attn==2.3.5" --no-build-isolation
```

以前は `flash-attn` と `torch` のバージョン乖離があり保留していたが、この組み合わせでは利用できた。

## 動作確認と再現確認

### smoke test の感触

元メモ時点では「多分回っている」という状態までは到達している。
以後は、単に起動するかではなく、再現実験できていることの保証を取りにいく段階。

### 確認観点

観点は大きく 2 つ。

1. 学習ログ上で loss が下がっており、学習が正常に進んでいるか
2. ベンチマークが論文どおりの傾向で出るか

### eval について

`scripts` 配下にある eval 用の shell script を実行すれば、`eval` の answer 出力の中に評価結果が入るらしい。
少なくとも eval の導線はありそうなので、次は学習ロギングと学習時間見積もりを詰めるのがよい。

## 優先度つき TODO

- `stage2` チェックポイントと CLIP 重みを配置して、学習スクリプトの参照先を固定する
- `MoE-LLaVA` データセットの展開フローを 1 本のスクリプトにまとめる
- smoke test 用の DeepSpeed 設定を `zero2.json` に寄せる
- `model_max_length` を最低 `512`、必要なら `2048` に上げる
- `flash-attn` が有効な環境を固定する
- loss logging と eval 出力で「再現できている」と言える条件を明文化する

## 補足メモ

- README のコマンドは、先頭に `uv` を付ければ概ねそのまま回せそう
- `hf download` を使う前提で進めた方がデータ取得は楽
- データ置き場は、ローカルコンテナ内に持つ方法と RAID を mount する方法の両方を検討できる
