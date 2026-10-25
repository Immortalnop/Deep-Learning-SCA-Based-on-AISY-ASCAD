# Console + file logging shared by the phase scripts.
#
# Usage:
#     with log_to(os.path.join(folder, "log.txt")):
#         ...  # everything printed here also goes to log.txt, errors included
import sys
import traceback
from contextlib import contextmanager


class Tee:
    # Writes to the console and the log file at the same time.
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def log_to(path):
    log_file = open(path, "w", encoding="utf-8")
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(stdout, log_file), Tee(stderr, log_file)
    try:
        yield
    except BaseException:
        # Python prints the traceback to the console; also keep it in the log.
        traceback.print_exc(file=log_file)
        raise
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        log_file.close()
