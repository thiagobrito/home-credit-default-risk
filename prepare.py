from prepare_application import *
from prepare_bureau_balance_data import *
from prepare_bureau_data import *
from prepare_previous_applications import *
from prepare_pos_cash import *
from prepare_installments_payments import *
from prepare_credit_card_balance import *

if __name__ == '__main__':
    prepare_bureau_balance()
    prepare_bureau_data()
    prepare_previous_applications()
    installments_payments()
    prepare_pos_cash()
    prepare_credit_card_balance()
    prepare_application()
