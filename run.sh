## Change Jack to Simon using mixed precision training and prior preservation
# python illustrator.py \
#     --orig_name Jack \
#     --custom_name Simon \
#     --custom_img_dir "sample_images/simon_512" \
#     --prompts_path output/jack_and_the_beanstalk/story.txt \
#     --config_path "config/text-encoder-mixed-precision-prior-preserve.yml" \
#     --device cuda:0 \

## change jack to simon but with generic orig_name, specific custom_name, and generic character description
python illustrator.py \
    --orig_object "boy" \
    --orig_name "Jack" \
    --custom_name Simonstasta \
    --custom_img_dir "sample_images/simon_512" \
    --prompts_path output/jack_and_the_beanstalk/story.txt \
    --config_path "config/text-encoder-mixed-precision-prior-preserve.yml" \
    --device cuda:0 \
    --skip_training

## load a previously trained dreambooth model
# python illustrator.py \
    # --orig_name Jack \
    # --prompts_path output/jack_and_the_beanstalk/story.txt \
    # --config_path "config/dreambooth-sd1-5.yml" \
    # --device cuda:0 \
    # --model_id "runs/dreambooth-model"


## mixed precision inference using model without mixed precision: works very well without error
# python illustrator.py \
#     --orig_name Jack \
#     --prompts_path output/jack_and_the_beanstalk/story.txt \
#     --prefix "mdjrny-v4 kids illustration showcasing the story of 'Jack and the Beanstalk'" \
#     --suffix "drawn by Rebecca Sugar, bright engaging children's illustration, digital painting, big eyes, beautiful shading, beautiful colors, amazon kdp, happy, interesting, 2D" \
#     --config_path "config/text-encoder-mixed-precision.yml" \
#     --device cuda:0 \
#     --model_id "runs/dreambooth-model"

# load mixed precision trained model
# python illustrator.py \
#     --orig_name Jack \
#     --prompts_path output/jack_and_the_beanstalk/story.txt \
#     --prefix "mdjrny-v4 kids illustration showcasing the story of 'Jack and the Beanstalk'" \
#     --suffix "drawn by Rebecca Sugar, bright engaging children's illustration, digital painting, big eyes, beautiful shading, beautiful colors, amazon kdp, happy, interesting, 2D" \
#     --config_path "config/text-encoder-mixed-precision.yml" \
#     --device cuda:0 \
#     --model_id "runs/text-encoder-mixed-precision"
    