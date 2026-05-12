import datetime
import sys
from pathlib import Path

class LoggerTee:
    """
    Duplica la salida (stdout/stderr) hacia consola y archivo,
    añadiendo marca temporal a cada línea.
    """
    def __init__(self, log_path, stream):
        self.terminal = stream
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        if message.strip():  # evita líneas vacías
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            formatted = f"{timestamp} {message}"
        else:
            formatted = message

        # Escribir en ambos sitios
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(formatted)
        self.log.flush()

    def flush(self):
        # Requerido para compatibilidad con sys
        self.terminal.flush()
        self.log.flush()

    def fileno(self):
        # Requerido para servicios como vLLM
        return self.terminal.fileno()
    
    def isatty(self):
        # Requerido para servicios como vLLM
        return self.terminal.isatty()

    def close(self):
        self.log.close()


def enable_stdout_logging(log_file: Path):
    """
    Redirige stdout y stderr a LoggerTee.
    Se asegura de cerrar el archivo al terminar el proceso.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("")

    sys.stdout = LoggerTee(log_file, sys.stdout)
    sys.stderr = LoggerTee(log_file, sys.stderr)