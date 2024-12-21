#!/bin/bash

####### Git bashで実行　##########

# 転送するファイルのリスト
REMOTE_FOLDERS=(
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPG_SAC_240907/Log/240917_150210
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPG_SAC_240907/Log/240917_150240
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPG_SAC_240907/Log/240917_151044
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPG_SAC_240907/Log/240921_171819
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPG_SAC_240907/Log/240921_172331
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/240927_213834
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/240927_213840
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241004_205243
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241004_205252
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241004_205318
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241004_214420
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241009_200312
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241008_124938
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241009_200058
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241009_200312
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241008_124951
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241011_132218
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPG_SAC_240907/Log/241017_210152
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPG_SAC_240907/Log/241017_210144
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPG_SAC_240907/Log/241018_203811
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPG_SAC_240907/Log/241016_220228
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241022_014030
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241028_003008
    # /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPG_SAC_240907/Log/241021_173827
    /mnt/ssd1/ryosei/master/A1_WS_RL4Real3/Scripts/A1CPGResiduals_SAC_240910/Log/241031_170145

)

# 設定
SSH_KEY_PATH="C:/Users/hayas/.ssh/id_ed25519"
REMOTE_USER="ryosei"
REMOTE_HOST="10.240.77.20"

for REMOTE_FOLDER in "${REMOTE_FOLDERS[@]}"; do
    # Log以下のパスを抽出
    RELATIVE_PATH=${REMOTE_FOLDER#*/Scripts/}
    echo $RELATIVE_PATH

    # ローカルディレクトリに追加
    # LOCAL_DIR="/mnt/c/Users/hayas/workspace_A1Real/Log2/${RELATIVE_PATH%/*}"
    LOCAL_DIR="C:/Users/hayas/A1Real/NN/${RELATIVE_PATH}"

    # ローカルディレクトリ作成
    mkdir -p "${LOCAL_DIR}"
    mkdir -p "${LOCAL_DIR}/Networks"

    # 最大の episode_ ファイルを見つける
    MAX_EPISODE_FILE=$(ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} "ls ${REMOTE_FOLDER}/Networks/episode_*.pt | sort -V | tail -n 1")
    # MAX_EPISODE_FILE="${REMOTE_FOLDER}/Networks/episode_4900.pt"
    #拡張子のないファイル名を取得
    # 拡張子のないファイル名を取得
    # NO_EXT_FILES=$(ssh -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} "find ${REMOTE_FOLDER} -type f ! -name '*.*'")

    # ファイル転送
    scp -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FOLDER}/config.json ${LOCAL_DIR}
    scp -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST}:${MAX_EPISODE_FILE} ${LOCAL_DIR}/Networks
    # scp -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST}:${NO_EXT_FILES} ${LOCAL_DIR}
done