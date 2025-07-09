import os, sys
import logging, platform


def create_logs(log_name, filename, log_message, status='info'):
    """
    Create a log file with the specified name and message.
    Args:
        log_name (str): Name of the logger.
        filename (str): Name of the log file (without extension).
        log_message (str): Message to log.
        status (str): Log level ('info', 'error', 'warning', 'debug').
    """
    
    if status == "console":
        print(log_message)
        return  # Exit if only console output is needed
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    foldername = "logs"
    if not os.path.exists(foldername):
        foldername = os.path.join(os.getcwd(), foldername)
        if not os.path.exists(foldername):
            os.makedirs(foldername)
    else:
        foldername = os.path.abspath(foldername)
        
    filename = filename if filename.endswith('.log') else filename + '.log'
    full_filename = os.path.join(foldername, filename)
    

    # Check if logs directory exists, if not create it
    try:
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            print(f"Created logs directory: {foldername}")
        elif not os.path.isdir(foldername):
            # If 'logs' exists but is not a directory (e.g., it's a file)
            raise OSError(f"'{foldername}' exists but is not a directory")
    except OSError as e:
        print(f"Error creating logs directory: {e}")
        return  # Exit function if we can't create the directory

    # Create a logger
    logger = logging.getLogger(log_name)
    # Set logger to debug level to catch all messages
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Check if logger already has handlers to prevent duplicate messages
    if not logger.handlers:
        try:
            # Create file handler
            file_handler = logging.FileHandler(full_filename)
            file_handler.setFormatter(formatter)

            # Create stream handler for terminal output
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)

            # Set the level for handlers based on the status
            if status == 'info':
                file_handler.setLevel(logging.INFO)
                stream_handler.setLevel(logging.INFO)
            elif status == 'error':
                file_handler.setLevel(logging.ERROR)
                stream_handler.setLevel(logging.ERROR)
            elif status == 'warning':
                file_handler.setLevel(logging.WARNING)
                stream_handler.setLevel(logging.WARNING)
            elif status == 'debug':
                file_handler.setLevel(logging.DEBUG)
                stream_handler.setLevel(logging.DEBUG)

            # Add handlers to the logger
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
        except Exception as e:
            print(f"Error setting up logger handlers: {e}")
            return

    # Log the message
    try:
        if status == 'info':
            logger.info(log_message)
        elif status == 'error':
            logger.error(log_message)
        elif status == 'warning':
            logger.warning(log_message)
        elif status == 'debug':
            logger.debug(log_message)
    except Exception as e:
        print(f"Error logging message: {e}")
