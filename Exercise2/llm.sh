#!/bin/bash
#PBS -q lecture-g
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -W group_list=gt00
#PBS -j oe

cd $PBS_O_WORKDIR
cd "$(pwd -P)"
module load apptainer

set -e

PORT=8000
echo "Start"

# =========================================
# 1. vLLM
# =========================================
apptainer exec --nv --env XDG_CACHE_HOME=cache vllm.sif \
	vllm serve models/Qwen3.8-27B \
		--served-model-name qwen3-coder\
		--host 127.0.0.1 --port ${PORT} \
		--enable-auto-tool-choice \
		--tool-call-parser qwen3_coder \
		--max-num-seqs 64 \
		> workspace/vLLM.log 2>&1 &

VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

# =========================================
# 2. vLLM 起動待ち
# =========================================

echo "Waiting for vLLM..."

until curl -sf \
    http://127.0.0.1:${PORT}/v1/models \
    > /dev/null; do
    sleep 5
done

echo "vLLM is ready."

# =========================================
# 3. OpenCode 設定
# =========================================

export OPENCODE_CONFIG_CONTENT='{
	"provider": {
    	"local": {
    		"npm": "@ai-sdk/openai-compatible",
    		"options": {
    			"baseURL": "http://127.0.0.1:8000/v1"
    		},
    		"models": {
    			"qwen3-coder": {}
    		}
    	}
    }
}'

# =========================================
# 4. OpenCode 自律実行
# =========================================

APPTAINER_ROOT=$(dirname "$(dirname "$(which apptainer)")")
apptainer exec \
    --bind workspace:/workspace \
    --pwd /workspace \
    --env OPENCODE_CONFIG_CONTENT="${OPENCODE_CONFIG_CONTENT}" \
    pytorch-opencode.sif \
    opencode run \
        --model local/qwen3-coder \
        "$(cat workspace/prompt.txt)" \
        > workspace/OpenCode.log 2>&1


# =========================================
# 5. vLLM 終了
# =========================================

kill ${VLLM_PID}
wait ${VLLM_PID} 2>/dev/null || true

trap - EXIT

echo "Finished."




# =========================================
# 以下、インタラクティブジョブでの実行方法
# =========================================

# # 計算ノードへログインし、ノードの hostname を取得
# qsub -I -l select=1 -W group_list=gt00 -q interact-g
# hostname
# # 計算ノード 内で、 vllm.sif コンテナの vllm を実行
# module load apptainer
# cd $PBS_O_WORKDIR
# apptainer exec --nv --env XDG_CACHE_HOME=cache vllm.sif \ 
# vllm serve models/Qwen3.8-27B --served-model-name qwen3-coder \
# --host 0.0.0.0 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
# --max-num-seqs 64

# # 新たに別のターミナルを起動し、 Miyabi-G へログイン。以降そのターミナルで実行。
# # 新たなターミナルから mgXXXX へ curl コマンドを実行し、返るか確認
# NODE_NAME=mgXXXX
# curl http://${NODE_NAME}:8000/v1/models
# # その別ターミナル上で OpenCode の変数を設定し、実行
# OPENCODE_CONFIG_CONTENT='{
# 	"$schema": "https://opencode.ai/config.json",
# 	"providers": {
# 		"local": { 
# 			"name": "Local vLLM",
# 			"package": "@opencode-ai/ai/providers/openai-compatible",
# 			"settings": {
# 				"baseURL": "http://$NODE_NAME:8000/v1"
# 				"api"
# 			},
# 			"models": {
# 				"qwen3-coder": {
# 					"modelID": "qwen3-coder",
# 					"name": "Qwen3 Coder"
# 				}
# 			}
# 		}
# 	}
# }'
# apptainer --env OPENCODE_CONFIG_CONTENT="$OPENCODE_CONFIG_CONTENT" opencode.sif opencode 
