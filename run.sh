# python illustrator.py \
#     --orig_name "Little Red Riding Hood" \
#     --prompts_path output/little_red_riding_hood/story.txt \
#     --prefix "mdjrny-v4 kids story illustration" \
#     --suffix "drawn by Rebecca Sugar, bright engaging children's illustration, digital painting, big eyes, beautiful shading, beautiful colors, amazon kdp, happy, interesting, 2D" \

# Change Jack to Simon
# python illustrator.py \
#     --orig_name Jack \
#     --custom_name Simon \
#     --custom_img_dir "sample_images/simon_512" \
#     --prompts_path output/jack_and_the_beanstalk/story.txt \
#     --prefix "mdjrny-v4 kids illustration showcasing the story of 'Jack and the Beanstalk'" \
#     --suffix "drawn by Rebecca Sugar, bright engaging children's illustration, digital painting, big eyes, beautiful shading, beautiful colors, amazon kdp, happy, interesting, 2D" \
#     --config_path "config/dreambooth-sd1-5.yml" \
#     --device cuda:0 \

# load a previously trained dreambooth model
python illustrator.py \
    --orig_name Jack \
    --prompts_path output/jack_and_the_beanstalk/story.txt \
    --prefix "mdjrny-v4 kids illustration showcasing the story of 'Jack and the Beanstalk'" \
    --suffix "drawn by Rebecca Sugar, bright engaging children's illustration, digital painting, big eyes, beautiful shading, beautiful colors, amazon kdp, happy, interesting, 2D" \
    --config_path "config/dreambooth-sd1-5.yml" \
    --device cuda:0 \
    --model_id "runs/dreambooth-model"

# python -m customization.dreambooth --train_batch_size 1  --max_train_steps 100 \
#         --enable_xformers_memory_efficient_attention 