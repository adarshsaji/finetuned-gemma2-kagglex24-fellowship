import os
from dotenv import load_dotenv
from groq import Groq
import groq
import json
import csv
import time
import logging
import io
from pathlib import Path
from ratelimit import limits, sleep_and_retry

load_dotenv()

# Constants
MAX_CALLS_PER_MINUTE = 30
SECONDS_PER_MINUTE = 60
INPUT_FILE = "Techcrunch_2023-10-24-to-2024-10-24.json"
OUTPUT_FILE = "qa_pairs_output.csv"
LOG_FILE = "process.log"
INDEX_FILE = "last_processed_index.log"
MODEL_INDEX_FILE = "current_model_index.log"
MAX_RETRIES = 3
MODELS = [
    "llama3-groq-70b-8192-tool-use-preview",
    "llama-3.2-90b-text-preview",
    "llama3-70b-8192",
    "llama-3.2-11b-text-preview",
    "llama3-groq-8b-8192-tool-use-preview",
]
RATE_LIMIT_PAUSE = 1200  # 20 minutes in seconds
MAX_CONSECUTIVE_429 = 10  # Maximum consecutive 429 errors before pausing

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

class StringIOHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.stream = io.StringIO()

    def emit(self, record):
        msg = self.format(record)
        self.stream.write(msg + '\n')

    def get_contents(self):
        return self.stream.getvalue()

    def clear(self):
        self.stream.truncate(0)
        self.stream.seek(0)

# Create and add the custom handler
string_handler = StringIOHandler()
logger.addHandler(string_handler)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Add this global variable after the constants
consecutive_429_count = 0

@sleep_and_retry
@limits(calls=MAX_CALLS_PER_MINUTE, period=SECONDS_PER_MINUTE)
def rate_limited_api_call(messages, model, max_tokens=8192, temperature=0.2):
    global consecutive_429_count
    
    try:
        response = client.chat.completions.create(messages=messages, model=model, max_tokens=max_tokens, temperature=temperature)
        consecutive_429_count = 0  # Reset only on successful response
        time.sleep(SECONDS_PER_MINUTE / MAX_CALLS_PER_MINUTE)
        return response
        
    except Exception as e:
        if "429" in str(e) or "rate_limit_exceeded" in str(e).lower():
            consecutive_429_count += 1
            logger.warning(f"Rate limit error in API call. Consecutive count: {consecutive_429_count}")
            
            if consecutive_429_count >= MAX_CONSECUTIVE_429:
                logger.warning(f"Hit maximum consecutive rate limits. Pausing for {RATE_LIMIT_PAUSE/60} minutes...")
                time.sleep(RATE_LIMIT_PAUSE)
                consecutive_429_count = 0
            
            raise groq.RateLimitError(str(e))
        raise e

def generate_qa_pairs(title, passage, model_index):
    global consecutive_429_count
    
    # Check payload size before making the API call
    prompt = f"""Given the title '{title}' and the following passage, generate 4 insightful question-answer pairs:

Passage: {passage}

Task: Generate 4 questions and their corresponding answers that will help entrepreneurs and startup founders gain valuable insights from this passage. The questions and answers should assist in ideation, validation, refinement of startup ideas, and identification of potential challenges or opportunities.

Criteria for questions:
1. Must be directly answerable from the given passage
2. Should be thought-provoking and encourage critical thinking about startup opportunities or challenges
3. Focus on key business insights, market trends, technological advancements, or industry-specific information
4. Avoid generic queries; instead, aim for specificity that could lead to actionable startup ideas
5. Do not use phrases like "according to the passage" or "in this context" - the questions should stand alone
6. Ensure questions are diverse, covering different aspects of startup development (e.g., market analysis, product development, funding strategies, potential obstacles)

Criteria for answers:
1. Provide comprehensive responses using only the information available in the passage
2. Highlight specific facts, figures, or examples that are relevant to startup development
3. Explain the significance of the information in the context of creating or growing a startup
4. If applicable, mention potential applications or implications for different industries or business models
5. Keep the tone informative and objective, avoiding speculation beyond the given information

Format your response as follows:
Q1: [Insightful question related to startup opportunities or challenges]
A1: [Comprehensive answer providing valuable information for entrepreneurs]

Q2: [Another insightful question focusing on a different aspect of startup development]
A2: [Detailed answer with relevant facts and implications for startups]

Q3: [Question addressing potential market trends or technological advancements]
A3: [Answer explaining the significance and potential impact on startups]

Q4: [Question about possible obstacles or critical considerations for startups in this domain]
A4: [Answer outlining key points and their relevance to startup success or failure]

Remember, each question-answer pair should provide unique, valuable insights that could spark ideas or guide decision-making for startup founders.
"""

    try:
        response = rate_limited_api_call([
            {"role": "system", "content": "You are an AI that generates question-answer pairs based on given titles and passages."},
            {"role": "user", "content": prompt}
        ], model=MODELS[model_index])
        
        content = response.choices[0].message.content.strip()
        pairs = []
        current_pair = {}
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('Q') and ':' in line:
                if current_pair:
                    pairs.append(current_pair)
                    current_pair = {}
                current_pair['question'] = line.split(':', 1)[1].strip()
            elif line.startswith('A') and ':' in line:
                current_pair['answer'] = line.split(':', 1)[1].strip()
        
        if current_pair:
            pairs.append(current_pair)
        
        return pairs, model_index

    except groq.RateLimitError as e:
        logger.warning(f"Rate limit reached in generate_qa_pairs. Error: {e}")
        # Switch models while keeping the rate limit counter
        return [], (model_index + 1) % len(MODELS)
    except Exception as e:
        error_message = str(e)
        if "413" in error_message or "400" in error_message or "payload too large" in error_message.lower():
            logger.warning(f"Payload too large for article. Skipping to next article.")
            return None, model_index
        else:
            logger.error(f"Unexpected error generating QA pairs: {e}")
            return [], (model_index + 1) % len(MODELS)

def save_progress(index):
    with open(INDEX_FILE, 'w') as f:
        f.write(str(index))

def save_model_index(index):
    with open(MODEL_INDEX_FILE, 'w') as f:
        f.write(str(index))

def load_model_index():
    if Path(MODEL_INDEX_FILE).exists():
        with open(MODEL_INDEX_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def main():
    start_index = 0
    if Path(INDEX_FILE).exists():
        with open(INDEX_FILE, 'r') as f:
            start_index = int(f.read().strip())
    
    model_index = load_model_index()
    
    logger.info(f"Starting/Resuming from index {start_index} with model {MODELS[model_index]}")

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    total_articles = len(data)

    with open(OUTPUT_FILE, mode="a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Instruction", "Response"], quoting=csv.QUOTE_MINIMAL)
        if start_index == 0:
            writer.writeheader()

        index = start_index
        while index < total_articles:
            article = data[index]
            
            qa_pairs, new_model_index = generate_qa_pairs(article.get("title", ""), article.get("passage", ""), model_index)
            
            if qa_pairs is None:  # Payload too large, skip to next article
                logger.info(f"Skipped article {index + 1} due to payload size")
                save_progress(index + 1)
                index += 1
                continue
            
            if not qa_pairs:  # Empty list means rate limit or error, retry same article
                model_index = new_model_index
                save_model_index(model_index)
                continue  # Retry same index
            
            try:
                for pair in qa_pairs:
                    writer.writerow({"Instruction": pair["question"], "Response": pair["answer"]})
                csvfile.flush()
                save_progress(index + 1)
                index += 1  # Only increment index on successful processing
            except KeyError as e:
                logger.warning(f"KeyError in article {index + 1}: {str(e)}. Skipping this article.")
                save_progress(index + 1)
                index += 1
                continue

            # Update model index if it has changed
            if new_model_index != model_index:
                logger.info(f"Switching to model: {MODELS[new_model_index]}")
                model_index = new_model_index
                save_model_index(model_index)

            logger.info(f"Processed article {index + 1} of {total_articles}")

    logger.info("Processing complete.")

if __name__ == "__main__":
    main()