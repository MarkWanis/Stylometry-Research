import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Tuple
import json
from datetime import datetime
import shutil
from tqdm import tqdm
import os


"""
Set up logging for the pipeline. 
This function creates a logger that logs messages to both a file and the console. 
The log file is stored in the specified log directory, and it rotates when it reaches a certain size. 
The logger is configured to log messages at different levels for the file and console.

log_dir: The directory where the log file will be stored. Defaults to "logs".
Returns: A logging.Logger object that can be used to log messages throughout the pipeline.
"""
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    
    # Make the logging directory if it isn't already made
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the logger (what is returned by the function)
    logger = logging.getLogger("StylometryPipeline")
    # By default the level of the logger is at debug
    logger.setLevel(logging.DEBUG)
    # This sets the format of each log, which maybe we would want to change in the future, but I can't imagine a need to
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    # This sets the max size of the log file before it starts overwriting old logs
    fh = RotatingFileHandler(log_dir / 'pipeline.log', maxBytes=10*1024*1024, backupCount=5)
    """The level of the file which determines what logs are saved"""
    fh.setLevel(logging.DEBUG)
    # Set the format that was set earlier
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    """The level of the console which determines what logs are shown in the terminal"""
    ch.setLevel(logging.INFO)
    # Set the format that was set earlier
    ch.setFormatter(formatter)
    
    # Add the file and console handlers to the logger so the logs are actually stored or shown
    logger.addHandler(fh)
    logger.addHandler(ch) 
    
    return logger

"""
Split text into chunks based on token count, handling long sequences safely. 

text: The input text to be chunked.
tokenizer: The tokenizer used to encode the text into tokens.
logger: The logger used for logging warnings and errors.
chunk_size: The maximum number of tokens allowed in each chunk. Defaults to 512.
Returns: A list of chunked text.
"""
def chunk_text(text: str, tokenizer, logger, chunk_size: int = 512) -> list:
    chunks = []

    # rough_chunks splits the text into sentences
    rough_chunks = [p for p in text.split('. ') if p.strip()]
    
    # current_chunk and current_length are used for having chunks be more than one sentence (used later in the function)
    current_chunk = []
    current_length = 0
    
    # try to tokenize every sentence (I had never seen tqdm before, but it just creates a progress bar for 'for' loops which is pretty cool)
    for rough_chunk in tqdm(rough_chunks, desc="Processing text chunks", leave=False):
        try:
            # The extracted text from the corpus keeps the newlines, resulting in linebreaks mid sentence. So to have the rough_chunks to just be based on the sentence, I replaced the newlines with a space
                # In the future, we may want to reconsider this, if the newlines could potentially have stylistic importance
                # (for example the corpus retains the line structure of the original work, then maybe for some works like poetry this would be important)
            rough_chunk = rough_chunk.replace("\n", " ")
            tokens = tokenizer.encode(rough_chunk + '.', add_special_tokens=False)

        # If the tokenization fails 
        except Exception as e:
            logger.warning(f"Error tokenizing chunk, splitting further: {str(e)}")
            # Split the sentence into words
            words = rough_chunk.split()

            tokens = []
            current_piece = []
            
            # For each word
            for word in words:
                # Create a list that will be tokenized (smaller than the original list)
                current_piece.append(word)
                # When the list is greater than 200 characters (which is smaller than 512, so it won't be too long)
                if len(' '.join(current_piece)) > 200: 
                    # Attempt to tokenize the smaller chunk
                    try:
                        piece_tokens = tokenizer.encode(' '.join(current_piece) + '.', add_special_tokens=False)
                        # Create a list of all the tokens in the sentence
                        tokens.extend(piece_tokens)
                        current_piece = []
                    # If that still failed, then now it is just discarded
                    except Exception as e:
                        logger.error(f"Could not tokenize piece: {str(e)}")
                        current_piece = []
                        continue
            # After all the words, tokenize what remains of the list (since it was likely the last word wouldn't put it over 200 characters)
            if current_piece:
                try:
                    piece_tokens = tokenizer.encode(' '.join(current_piece) + '.', add_special_tokens=False)
                    tokens.extend(piece_tokens)
                except Exception as e:
                    logger.error(f"Could not tokenize final piece: {str(e)}")
        
        # current_length is the length of the chunk after the last sentence's tokens were added
        # current_chunk is the chunk after adding the tokens of the last sentence

        # If adding this sentence to the chunk would make it be larger than the max size
        if current_length + len(tokens) > chunk_size and current_chunk:
            # Convert the chunk back into text (the previous chunk before adding the new sentence)
            chunk_text = tokenizer.decode(current_chunk)
            # Not quite sure what the strip function is doing here (maybe ensuring that an empty chunk of just spaces doesn't get added to the list)
            if chunk_text.strip():
                # Add the current chunk to the list that is returned
                chunks.append(chunk_text.strip())
            current_chunk = []
            current_length = 0
        
        # If the current sentence is bigger than the max size
        if len(tokens) > chunk_size:
            # This part isn't done too cleanly, just split it up into exactly 512 token parts
            for i in range(0, len(tokens), chunk_size):
                sub_tokens = tokens[i:i + chunk_size]
                # Decode the tokens
                sub_text = tokenizer.decode(sub_tokens)
                # And then add the chunk to the list that is returned
                if sub_text.strip():
                    chunks.append(sub_text.strip())
        
        # Otherwise, it is within the maximum size, so just add it to the current_chunk list and move onto the next chunk
        else:
            current_chunk.extend(tokens)
            current_length += len(tokens)
        
        # Okay, I'm kind of lost on the purpose of this if statement, as the current length wouldn't bigger than the max size due to the two previous if statements
        # If there is just a case I'm not thinking of, then if the length of the chunk is greater than the current size
        if current_length >= chunk_size:
            # Decode the tokens
            chunk_text = tokenizer.decode(current_chunk)
            # And add the text to the list that is returned
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            current_chunk = []
            current_length = 0
    
    ## Leaving the big for loop of going through each sentence

    # For the last sentence, if it wasn't already added to the list
    if current_chunk:
        # Decode it
        chunk_text = tokenizer.decode(current_chunk)
        # And add it to the list that is returned
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    

    # To display all of the tokens (used for presentation)
    #for c in chunks:
    #    logger.info(f"Chunk: {c}")

    # The list of chunks is returned
    return chunks


"""
This class is the main pipeline for processing text files and generating embeddings. 
It handles reading input files, chunking the text, generating embeddings using a pre-trained model, and saving the results. 
It also manages logging and error handling throughout the process.
"""
class StylometryPipeline:
    """
    Initialize the StylometryPipeline with specified directories for input, processed files, error files, and embedding results.
    Sets up logging, creates necessary directories, and loads the NLP models for tokenization and embedding generation.

    input_dir: Directory where input text files are located. Defaults to "data/input".
    processed_dir: Directory where successfully processed files will be moved. Defaults to "data/processed".
    error_dir: Directory where files that encountered errors during processing will be moved. Defaults to "data/error".
    results_dir: Directory where embedding results will be saved. Defaults to "data/embedding_results".
    """
    def __init__(self, input_dir: str = "data/input", 
                 processed_dir: str = "data/processed",
                 error_dir: str = "data/error",
                 results_dir: str = "data/embedding_results"):
        
        # The first step is setting up the logger (full function shown below, but mainly it just )
        self.logger = setup_logging()
        
        # Set up directories
        self.input_dir = Path(input_dir)
        self.processed_dir = Path(processed_dir)
        self.error_dir = Path(error_dir)
        self.results_dir = Path(results_dir)
        
        # If the directories don't already exist, then make new ones
        for dir_path in [self.input_dir, self.processed_dir, self.error_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ML components
        # This checks if it can run on GPU to go faster
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")


        self.logger.info("Loading NLP models...")
        # tokenizer is set up to encode and decode the tokens
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        # The model is then set up to later create the embeddings
        self.model = AutoModel.from_pretrained('AIDA-UPM/star').to(self.device)
        # The .eval() is setting the model to evaluation mode (which is different than if we wanted to train the model)
        self.model.eval()

    """
    Process a single chunk of text and return its embedding.

    chunk: The input text chunk to be processed.
    Returns: A numpy array representing the embedding of the input chunk.
    """
    def process_chunk(self, chunk: str) -> np.ndarray:
        try:
            # This is creating the tokens again, but this time it pads the tokens, so every input is exactly 512 tokens long (if it somehow was too long, it will also truncate it)
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # The torch.no_grad is turning off calculating the gradients, which looking into it, helps perform better when you don't need to use backpropagation (which we aren't training the model)
            with torch.no_grad():
                # https://huggingface.co/docs/transformers/v4.48.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput
                # This is what I've found so far to explain how this section works

                # This is giving the model the list of the tokens and creates a SequenceClassifierOutput
                outputs = self.model(**inputs)

                # This is then calculating the embeddings 
                # (the pooler_output means that it is going though a linear layer and tanh activation, which I need to read more about to better understand)
                embedding = outputs.pooler_output.cpu().numpy()
            # To my understanding, the pooler_output can do several chunks at the same time, so it returns a matrix, but we only want the first value, since we are only doing one embedding
            return embedding[0]
        # If something went wrong, then throw an error (I'm pretty sure this will stop the entire process_file function)
        except Exception as e:
            self.logger.error(f"Error processing chunk: {str(e)}")
            raise


    """
    Process a single text file with chunking.
    From start to finish, this function reads the text file, splits it into chunks, generates embeddings for each chunk, averages the embeddings, and saves the results to a JSON file.

    filepath: The path to the input text file.
    Returns: A tuple containing the job ID and the averaged embedding.
    """
    def process_file(self, filepath: Path) -> Tuple[str, np.ndarray]:
        self.logger.info(f"Processing file: {filepath}")
        
        try:
            # Read file     
            # The Corpus Texts are encoded with utf-8, the Bible with Windows-1252
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # Chunk text
            chunks = chunk_text(text, self.tokenizer, self.logger)
            self.logger.info(f"Split into {len(chunks)} chunks")

            chunk_dir = self.results_dir / 'chunks'
            chunk_dir.mkdir(parents=True, exist_ok=True)


            # Create job ID and metadata
            job_id = f"job_{filepath.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


            # Process each chunk
            chunk_embeddings = []
            for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
                # Create the embedding from the chunk
                embedding = self.process_chunk(chunk)
                # Add embedding to list of embeddings
                chunk_embeddings.append(embedding)

                # Save individual chunk embedding
                chunk_result = { job_id:
                    {
                        'chunk_text': chunk,
                        'embedding': embedding.tolist(),
                        'chunk_index': i
                    }
                }
                with open(chunk_dir / f'{filepath.stem}_chunk_{i}.json', 'w') as f:
                    json.dump(chunk_result, f, indent=2)

            # If there are no embeddings, then they can't be averaged
            if not chunk_embeddings:
                raise ValueError("No valid chunks processed")
            
            # Average embeddings
            final_embedding = np.mean(chunk_embeddings, axis=0)
            
            
            
            # Save results
            result = {
                job_id: {
                    'text': text,  # original text
                    'embedding': final_embedding.tolist(),
                    'metadata': {
                        'filename': filepath.name,
                        'processed_at': datetime.now().isoformat(),
                        'num_chunks': len(chunks),
                        'file_size': filepath.stat().st_size
                    }
                }
            }
            
            # Save to JSON
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Create the file in the embedding_results directory
            result_file = self.results_dir / f'embedding_{filepath.stem}_{timestamp}.json'
            
            # Store the data into the JSON file
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Once the data has been put into the file, it is then moved from embedding_results directory to processed directory
            dest_path = self.processed_dir / filepath.name
            shutil.move(str(filepath), str(dest_path))
            
            # The embeddings were all succesfully made, and the average embedding is stored in data/processed 
            self.logger.info(f"Successfully processed {filepath}")
            return job_id, final_embedding

        # If something did go wrong in processing the file, then raise an error, which will just move onto the next file
        except Exception as e:
            self.logger.error(f"Error processing file {filepath}: {str(e)}")
            error_path = self.error_dir / filepath.name
            shutil.move(str(filepath), str(error_path))
            raise


"""
Main function to run the stylometry pipeline.
This function initializes the StylometryPipeline and processes all text files in the specified input directory.
"""
def main():
    pipeline = StylometryPipeline()
    
    # For cases where the input files are all in seperate folders
    """
    for root, dirs, files in os.walk(pipeline.input_dir):
        for directory in dirs:
            d = Path(root) / directory
            for filepath in d.glob("*.txt"):
                try:
                    pipeline.process_file(filepath)
                except Exception as e:
                    pipeline.logger.error(f"Failed to process {filepath}: {str(e)}")
                    continue
    """

    # Process all txt files in input directory
    # By default this is in data/input, but can be changed to any directory with the first parameter of the StylometryPipleine constructor
    for filepath in pipeline.input_dir.glob("*.txt"):
        try:
            # Run the pipeline for that file    
            pipeline.process_file(filepath)
        except Exception as e:
            # If there's any error, then log it and move to the next file    
            pipeline.logger.error(f"Failed to process {filepath}: {str(e)}")
            continue

if __name__ == "__main__":
    main()