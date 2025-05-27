import sys
import traceback

class CustomException(Exception):
    def __init__(self, message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_number = exc_tb.tb_lineno
        self.message = f"Exception in {self.file_name}, line {self.line_number}: {message}"
        super().__init__(self.message)
    
    def __str__(self):
        return self.message
