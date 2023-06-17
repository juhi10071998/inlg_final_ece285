# inlg_final_ece285
This is the implementation code for Visually Augmented Text Generation Task, due to size limitations, we have not uploaded the code on github.

# Environment Setup
git clone https://github.com/juhi10071998/inlg_final_ece285.git
<br /> cd iNLG/
<br /> for dataset in activitynet commongen rocstories
<br /> do
<br />     mkdir -p log/${dataset}
<br /> done

# Step 2: Setup conda environment
<br /> conda env create -f env.yml
<br /> conda activate inlg
<br /> python -m spacy download en

## Text Generation
# For Concept2Text
bash scripts/run_concept2text_with_image.sh

# For Sentence Completion
bash scripts/run_sentence_completion_with_image.sh
