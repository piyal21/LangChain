from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import PyPDFLoader




splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0,
)



text = """

    class BankAccount:
    def __init__(self, account_holder, initial_balance=0.0):
        self.account_holder = account_holder
        self.balance = initial_balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"Deposited ${amount:.2f}. New balance is ${self.balance:.2f}")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if amount <= 0:
            print("Withdrawal amount must be positive.")
        elif amount > self.balance:
            print("Insufficient funds.")
        else:
            self.balance -= amount
            print(f"Withdrew ${amount:.2f}. New balance is ${self.balance:.2f}")

    def get_balance(self):
        return self.balance

"""


# --> splitting. 

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])
