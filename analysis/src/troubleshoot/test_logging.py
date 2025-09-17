import logging

def test_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test.log'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("This is an info message.")
    logging.error("This is an error message.")

if __name__ == "__main__":
    test_logging()
