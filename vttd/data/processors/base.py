class Processor:
    """
    Processor base class.
    """
    def process(self, text: str):
        """
        Process text.
        :param text:    Text to be processed.
        """
        raise NotImplementedError("Please implement this method in your processor class.")
