import os, sys
import logging, platform, json
from typing import Dict, List, Any

CURRENT_DIR = os.getcwd()
DATETIME_FT = '%d-%m-%Y %H:%M:%S'
DEBUG = True

class LocalizationManager:
    """
    A class to manage loading and retrieving internationalization (i18n) strings
    from JSON files for a PySide6 application.

    It uses a dot-notation key (e.g., 'MAIN_WINDOW.title') to access nested
    JSON values and caches loaded language files to avoid redundant disk I/O.
    """

    def __init__(self, locale_dir: str = 'assets/locales', default_lang: str = 'en', initial_lang: str = None):
        """
        Initializes the LocalizationManager.

        Args:
            locale_dir (str): The directory where locale JSON files (e.g., 'en.json') are stored.
            default_lang (str): The fallback language to use if a key is not found in the current language.
            initial_lang (str): The initial language to set. Defaults to `default_lang`.
        """
        self.locale_dir = locale_dir
        if not os.path.isdir(self.locale_dir):
            # Try to resolve path relative to the current file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.locale_dir = os.path.join(base_dir, '..', locale_dir)
            if not os.path.isdir(self.locale_dir):
                create_logs("LocalizationManager", "localization", f"Locale directory not found: {self.locale_dir}", status='error')
                raise FileNotFoundError(f"Locale directory not found: {self.locale_dir}")

        self.translations: Dict[str, Dict[str, Any]] = {}
        self.default_lang = default_lang
        self.current_lang = initial_lang if initial_lang else default_lang

        # Pre-load the default and initial languages
        self._load_language(self.default_lang)
        if self.current_lang != self.default_lang:
            self._load_language(self.current_lang)

    def _load_language(self, lang_code: str) -> bool:
        """
        Loads a language file from the locale directory.

        Args:
            lang_code (str): The language code (e.g., 'en', 'ja').

        Returns:
            bool: True if the language was loaded successfully, False otherwise.
        """
        if lang_code in self.translations:
            return True

        file_path = os.path.join(self.locale_dir, f"{lang_code}.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.translations[lang_code] = json.load(f)
                create_logs("LocalizationManager", "localization", f"Successfully loaded language: {lang_code}", status='info')
                return True
        except FileNotFoundError:
            create_logs("LocalizationManager", "localization", f"Language file not found: {file_path}", status='warning')
            return False
        except json.JSONDecodeError as e:
            create_logs("LocalizationManager", "localization", f"Error decoding JSON from {file_path}: {e}", status='error')
            return False

    def set_language(self, lang_code: str):
        """
        Sets the current language for translations.

        Args:
            lang_code (str): The language code to switch to.
        """
        if self._load_language(lang_code):
            self.current_lang = lang_code
        else:
            create_logs("LocalizationManager", "localization", f"Failed to set language to '{lang_code}'. It may not exist.", status='error')

    def get(self, key: str, **kwargs) -> str:
        """
        Retrieves a translated string using a dot-separated key.

        It first tries the current language, then falls back to the default language,
        and finally returns the key itself if not found.

        Args:
            key (str): The dot-separated key (e.g., 'PAGE_NAME.option').
            **kwargs: Values to format into the translated string.

        Returns:
            str: The translated and formatted string.
        """
        keys = key.split('.')

        # Try to get from the current language
        translation = self._find_translation(self.current_lang, keys)

        # If not found, try the default language
        if translation is None and self.current_lang != self.default_lang:
            translation = self._find_translation(self.default_lang, keys)

        # If still not found, return the key itself as a fallback
        if translation is None:
            create_logs("LocalizationManager", "localization", f"Translation key not found: '{key}'", status='warning')
            return keys[1].replace('_', ' ')

        # Format the string with any provided keyword arguments
        try:
            return translation.format(**kwargs)
        except KeyError as e:
            create_logs("LocalizationManager", "localization", f"Missing format argument {e} for key '{key}'", status='warning')
            return translation # Return unformatted string

    def _find_translation(self, lang_code: str, keys: List[str]) -> str | None:
        """
        A helper to traverse the dictionary for a given language.
        """
        if lang_code not in self.translations:
            return None

        data = self.translations.get(lang_code)
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return None

        return data if isinstance(data, str) else None

    def available_languages(self) -> List[str]:
        """
        Scans the locale directory and returns a list of available languages.

        Returns:
            List[str]: A list of language codes found in the directory.
        """
        try:
            files = os.listdir(self.locale_dir)
            # Return the filename without the .json extension
            return [f.split('.')[0] for f in files if f.endswith('.json')]
        except FileNotFoundError:
            return []

    def tr(self, key: str, **kwargs) -> str:
        """
        A convenient alias for the get() method.
        """
        return self.get(key, **kwargs)

def load_config(file_path: str = "configs/app_configs.json") -> dict:
    """
    Load a JSON configuration file and return its contents as a dictionary.
    
    Args:
        file_path (str): Path to the JSON configuration file.
        
    Returns:
        dict: Contents of the JSON file.
    """
    if not os.path.exists(file_path):
        create_logs("ConfigLoader", "config", f"Configuration file not found: {file_path}", status='error')
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            create_logs("ConfigLoader", "config", f"Successfully loaded configuration from {file_path}", status='info')
            return config
    except json.JSONDecodeError as e:
        create_logs("ConfigLoader", "config", f"Error decoding JSON from {file_path}: {e}", status='error')
        return {}

def create_logs(log_name="LOGS", filename="logs", log_message="", status='info', show_console=False):
    """
    Create a log file with the specified name and message.
    Args:
        log_name (str): Name of the logger.
        filename (str): Name of the log file (without extension).
        log_message (str): Message to log.
        status (str): Log level ('info', 'error', 'warning', 'debug').
    """
    
    if status == "console" and show_console:
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
        elif not os.path.isdir(foldername):
            # If 'logs' exists but is not a directory (e.g., it's a file)
            raise OSError(f"'{foldername}' exists but is not a directory")
    except OSError as e:
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
            # Create file handler with UTF-8 encoding to handle Unicode characters
            file_handler = logging.FileHandler(full_filename, encoding='utf-8')
            file_handler.setFormatter(formatter)

            # Create stream handler for terminal output with UTF-8 encoding
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            
            # Set stream encoding to UTF-8 if possible
            if hasattr(stream_handler.stream, 'reconfigure'):
                try:
                    stream_handler.stream.reconfigure(encoding='utf-8')
                except Exception:
                    pass  # Fallback to default encoding

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
        pass  # Silently ignore logging errors

def load_application_fonts():
    """
    Loads all .ttf fonts from the assets/fonts directory into the application.
    This makes them available for use in stylesheets without system installation.
    """
    from PySide6.QtGui import QFontDatabase
    # Try to resolve fonts directory relative to the project root if not found in current dir
    fonts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'fonts')
    if not os.path.exists(fonts_dir):
        # Fallback: try relative to this file (legacy location)
        fonts_dir = os.path.join(os.path.dirname(__file__), 'assets', 'fonts')
    if not os.path.exists(fonts_dir):
        return

    for font_file in os.listdir(fonts_dir):
        if font_file.endswith('.ttf'):
            font_path = os.path.join(fonts_dir, font_file)
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id == -1:
                pass  # Font loading failed
            else:
                # You can optionally print the loaded font families
                families = QFontDatabase.applicationFontFamilies(font_id)
                # print(f"Successfully loaded font: {families[0]}")
    
    # After loading, it's good practice to log all available families to debug
    # print("All available font families:", QFontDatabase().families())