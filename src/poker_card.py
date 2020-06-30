import numpy as np
l = np.arange(1, 22)
traget_card = np.random.randint(1, 22)
print(traget_card)
for _ in range(3):
	l_head = l[::3]
	l_middle = l[1::3]
	l_end = l[2::3]
	print(l_head)
	print(l_middle)
	print(l_end)
	
	if traget_card in l_head:
		l = np.hstack((l_middle, l_head, l_end))
	elif traget_card in l_middle:
		l = np.hstack((l_head, l_middle, l_end))
	else:
		l = np.hstack((l_head, l_end, l_middle))
	print(l)
print(l[10])