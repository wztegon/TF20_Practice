import numpy as np
all_cards = [i for i in range(2, 11)]*4 + ['J', 'Q', 'K', 'A'] * 4 + ['s_joker', 'l_JOKER']
all_cards *= 3
print(len(all_cards))
role_1 = []
role_2 = []
role_3 = []
role_4 = []
role_5 = []
print(all_cards)
while True:
	str_input = input("请输入： ")
	if(str_input == 'q'):
		break
	print(str_input)