# 安装项目所需的依赖包
pip install -r requirements.txt

# 创建数据目录
mkdir -p data/writingPrompts

# 下载writingPrompts数据集（使用wget和curl两种方式确保下载成功）
wget https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz
curl -O https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz

# 解压数据集到指定目录
tar -xzf writingPrompts.tar.gz \
    --strip-components=1 \
    -C data/writingPrompts \
    writingPrompts

# 下载并准备各种模型（用于检测任务）

#python scripts/model.py --model_name='gpt-j-6B'
#python scripts/model.py --model_name='gpt-neo-2.7B'
#python scripts/model.py --model_name='qwen-7b'
#python scripts/model.py --model_name='mistralai-7b'
#python scripts/model.py --model_name='llama3-8b'
#python scripts/model.py --model_name='falcon-7b'
#python scripts/model.py --model_name='falcon-7b-instruct'
#python scripts/model.py --model_name='gemma-9b'
#python scripts/model.py --model_name='gemma-9b-instruct'

# 设置NLTK数据目录
NLTK_DIR="$HOME/nltk_data"

# 创建NLTK所需的目录结构（已注释，可根据需要启用）
# mkdir -p "$NLTK_DIR/corpora"
# mkdir -p "$NLTK_DIR/tokenizers"

# 下载并安装NLTK停用词数据
wget -O "$NLTK_DIR/stopwords.zip" \
  https://github.com/nltk/nltk_data/raw/refs/heads/gh-pages/packages/corpora/stopwords.zip
unzip -o "$NLTK_DIR/stopwords.zip" -d "$NLTK_DIR/corpora/"
rm "$NLTK_DIR/stopwords.zip"

# 下载并安装NLTK WordNet数据
wget -O "$NLTK_DIR/wordnet.zip" \
  https://github.com/nltk/nltk_data/raw/refs/heads/gh-pages/packages/corpora/wordnet.zip
unzip -o "$NLTK_DIR/wordnet.zip" -d "$NLTK_DIR/corpora/"
rm "$NLTK_DIR/wordnet.zip"

# 下载并安装NLTK OMW-1.4数据（多语言WordNet）
wget -O "$NLTK_DIR/omw-1.4.zip" \
  https://github.com/nltk/nltk_data/raw/refs/heads/gh-pages/packages/corpora/omw-1.4.zip
unzip -o "$NLTK_DIR/omw-1.4.zip" -d "$NLTK_DIR/corpora/"
rm "$NLTK_DIR/omw-1.4.zip"

# 下载并安装NLTK punkt_tab分词器
wget -O "$NLTK_DIR/punkt_tab.zip" \
  https://github.com/nltk/nltk_data/raw/refs/heads/gh-pages/packages/tokenizers/punkt_tab.zip
unzip -o "$NLTK_DIR/punkt_tab.zip" -d "$NLTK_DIR/tokenizers/"
rm "$NLTK_DIR/punkt_tab.zip"

## 请将NLTK_DATA="$HOME/nltk_data"添加到环境变量中，以便程序能找到NLTK数据
## Please add NLTK_DATA="$HOME/nltk_data" into the environment path. 