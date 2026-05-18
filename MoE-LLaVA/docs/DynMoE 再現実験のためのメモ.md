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

setup_moellava.sh で入れるとエラーが出るけど、手動で入れれば通る。
なんかpypiへのリクエストが失敗してるっぽかった。ここら辺は注視しておくと良いかも。

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

## 学習ロギング方針メモ

### まず取るべき項目

最初の short run では、まず Hugging Face Trainer が標準で出せる項目を `wandb` に流し、
学習が正常に進んでいるかを確認する。

- `train/loss`
- `train/learning_rate`
- `train/epoch`
- `train/global_step`
- run 名
- checkpoint 保存 step

### 次に追加したい項目

full run の前に、次の項目も追加で取りたい。

- `step_time_sec`
- `samples_per_sec`
- `tokens_per_sec`
- `gpu_memory_allocated`
- `gpu_memory_reserved`
- `router_aux_loss`
- `expert_usage`
- `active_expert_count`
- `tokens_per_expert`
- `dead_expert_detected`
- `add_remove_event`
- `eta_hours`

### short run での確認観点

- `wandb` に run が作成されるか
- step 1 から loss が見えるか
- loss が数 step で極端に発散しないか
- learning rate が想定どおりに記録されるか
- local checkpoint / local dataset / flash-attention 経路が落ちずに通るか

## wandb を混ぜた short run 手順

### 事前条件

- `wandb` パッケージが入っていること
- `wandb login` が済んでいること
- `Stage2` checkpoint と dataset path が解決済みであること

### 実行例

`MoE-LLaVA` 配下で以下を実行する。

```bash
cd /usr/src/app/DynMoE/MoE-LLaVA
uv run wandb login
bash scripts/v2/stablelm/finetune_dynmoe_smoke.sh
```

補足:

- smoke script は `max_steps=3` なので、最初の接続確認に向いている
- `wandb` を確実に確認したいだけなら、最初は smoke で十分

### 確認手順

1. terminal 上で `trainer.train()` まで進み、step ログが出ることを確認する
2. `wandb` の UI で対象 project に run が作成されていることを確認する
3. `loss` と `learning_rate` が step ごとに記録されていることを確認する
4. run 名と run group が意図どおりであることを確認する
5. smoke が完走したら、次は save あり short run に進む

### 次の一手

smoke で `wandb` 連携が確認できたら、次は full 条件に近い short run を別 script で切る。
その run では少なくとも次を確認したい。

- `save_strategy=steps` で checkpoint が保存される
- `zero2_offload.json` 経路で落ちない
- `train_mem.py + flash-attention` で安定して数十 step 回る
- `wandb` 上で loss の傾向から full run の所要時間見積もりに必要な基礎ログが取れる

## Next Action Plan

このセクションは、現時点から full finetuning と評価接続まで進めるための実行順を、
そのまま作業に移せる粒度で固定したもの。

### Step 1. wandb 付き smoke 成功を記録する

目的:

- `train_mem.py + flash-attention + deepspeed + wandb` の疎通確認が完了した状態を固定する

やること:

- 実行日時をメモする
- 対象 run 名を記録する
- `wandb` 上で `loss` と `learning_rate` が見えていることを確認する
- `rank_logs/` の対象 run が全 rank success で終わっていることを確認する

完了条件:

- smoke run が完走
- `wandb` 上で標準学習ログが見える
- rank log に致命エラーがない

現状:
- smoke run 完走
- wandbでloss, samples/sec step/sec などが見える。いずれもtrain
- rank logに致命的なエラーはなく、本格的に回していないため他のGPUが休んで、GPU0が忙しそうにしているが問題はなさそう。

### Step 2. save あり short run 用 script を作る

目的:

- full 本番の前に checkpoint 保存と logging を確認できる短時間 run を作る

やること:

- `scripts/v2/stablelm/finetune_dynmoe_short.sh` を新規作成する
- ベースは `finetune_dynmoe.sh` を使う
- `max_steps` を 20 から 50 程度にする
- `save_strategy=steps` を有効にする
- `save_steps` を小さくする
- `output_dir` を smoke や full 本番と分ける
- `REPORT_TO=wandb` を既定にする

完了条件:

- short run 用 script が単体で実行できる
- output と run 名が他の実験と衝突しない

現状:
- `finetune_dynmoe_short.sh` を作成
- `max_steps=20`, `save_strategy=steps`, `save_steps=5` を設定
- `output_dir` を `./outputs/short_dynmoe` に設定
- zero2.json を指定。zero2_offload.json だとエラーが起こってしまったため。
- checkpoint-20というディレクトリができることを確認
  - この中にptファイル系の一連が入っていることを確認した。
- zero2_offload.jsonはpendingの判断をする。

### Step 3. resume を検証する

目的:

- 長時間学習前に復旧経路を保証する

やること:

- Step 2 の `output_dir` をそのまま使う
- 同じ script を再実行する
- `checkpoint-*` 自動検知で resume に入るか確認する
- resume 後も loss が不自然に跳ねないか確認する

完了条件:

- `resume_from_checkpoint=True` 経路が動く
- resume 後に学習が継続する

現状:
- resumeを明示的にTrueにせずとも、同じチェックポイントが入ったディレクトリでmax_stepを伸ばして実行すればそこからresumeされることがわかった。
- wandb上では別のrunとして記録されるが、横軸はresumeされた状態であったので成功。


### Step 4. 追加 logging 項目の実装方針を固める

目的:

- full run の所要時間見積もりと MoE 挙動監視に必要な指標を決める

やること:

- `wandb` に今すでに出ている項目を整理する
- 追加したい項目を優先度順に並べる
- まず callback で足す項目を決める

優先度高:

- `step_time_sec`
- `samples_per_sec`
- `tokens_per_sec`
- `gpu_memory_allocated`
- `gpu_memory_reserved`
- `eta_hours`

優先度中:

- `router_aux_loss`
- `expert_usage`
- `active_expert_count`
- `tokens_per_expert`

完了条件:

- 追加 logging 項目の実装順が決まっている
- どのファイルに callback / logging を入れるか決まっている

### Step 5. full run 前の go/no-go を判定する

目的:

- 本番の長時間実験に進んでよいかを明確に判断する

go 条件:

- smoke は完了済み
- save あり short run が完了済み
- resume が確認済み
- `wandb` の標準ログが安定して見える
- `zero2_offload.json` 経路で落ちていない

no-go 条件:

- checkpoint 保存が壊れている
- resume 後に loss が破綻する
- W&B run が途中で止まる
- rank log に OOM / NaN / recurrent error が出る

### Step 6. full finetuning を実行する

目的:

- StableLM DynMoE full finetuning を 1 回完走させる

やること:

- `finetune_dynmoe.sh` を本番設定で実行する
- `wandb` 上で loss 推移と step 進行を監視する
- checkpoint rotation と disk 使用量を確認する
- 必要なら保存間隔や log 頻度を微調整する

完了条件:

- full run が最後まで完走する
- 最終 checkpoint が出力される

### Step 7. 評価導線へ接続する

目的:

- 学習済み checkpoint を benchmark 評価につなぐ

やること:

- `docs/EVAL.md` に沿って `CKPT_NAME` を差し替える
- まず `ScienceQA` を回す
- 次に `MME`, `SEED` へ広げる
- 結果の保存先と比較表の形式を固定する

完了条件:

- 学習済み checkpoint から少なくとも 1 benchmark の評価が通る

### 実行順の要約

1. smoke 成功を記録する
2. save あり short run 用 script を作る
3. save あり short run を回す
4. resume を確認する
5. 追加 logging の実装方針を決める
6. full run の go/no-go を判定する
7. full finetuning を実行する
8. 評価へ接続する

