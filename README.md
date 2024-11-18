# Gemma Startup Engine  

The Gemma Startup Engine is a fine-tuned version of the `gemma2_2b_en` model, designed to assist with startup-related queries by providing insights and recommendations. Fine-tuned with Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Rank 4), the model leverages synthetic QA pairs generated from daily startup news (October 2023 - October 2024). The fine-tuning process utilised 27k trainable parameters, making it highly efficient and adaptable.  

## Features  
- **Startup Expertise**: Provides actionable insights for startup ideas, market opportunities, and business strategies.  
- **Efficient Training**: Fine-tuned using PEFT with LoRA Rank 4.
- **Synthetic QA Generation** : Synthetic QA generated using Llama 3.2 model hosted on Groq


## Quick Links  
- **Live Deployment**: Test the model interactively on [Hugging Face Spaces](https://huggingface.co/spaces/AdarshSaji/StartupBuddy).  
- **Model Repository**: Access the model on the [Kaggle Model Hub](https://www.kaggle.com/models/adarshsaji/gemma/keras/gemma-startup-engine).  

## Notes  
The model was fine-tuned on synthetic data and is not instruction-tuned, which may limit generalisation to highly domain-specific or sensitive queries.
