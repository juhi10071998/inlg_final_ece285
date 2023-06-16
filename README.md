# inlg_final_ece285
This is the implementation code for Visually Augmented Text Generation Task

# Environment Setup
git clone 
<br /> cd iNLG/
for dataset in activitynet commongen rocstories
do
    mkdir -p log/${dataset}
done

# Step 2: Setup conda environment
conda env create -f env.yml
conda activate inlg
python -m spacy download en

## Text Generation
# For Concept2Text
bash scripts/run_concept2text_with_image.sh

# For Sentence Completion
bash scripts/run_sentence_completion_with_image.sh
