L = []
moves = [(1, 3),    (0, 2, 4),    (1, 5),
         (0, 4, 6), (1, 3, 5, 7), (2, 4, 8),
         (3, 7),    (4, 6, 8),    (5, 7)]  # 每个位置的0可以交换的位置


def slidingPuzzle(board):
	board = board[0] + board[1] + board[2] # 把board连起来变一维
	
	q= [(tuple(board), board.index(0), 0)] # bfs队列和已访问状态记录
	visited = set()
	while q:
		if len(L) == 0:
			L.append(q[:])
		
		state, now, step = q.pop(0)  # 分别代表当前状态，0的当前位置和当前步数
		
		if L[-1][-1][-1] == step:
			L[-1].append((state, now, step))
		else:
			L.append([(state, now, step)])
		#print(L)
		if state == (1, 2, 3, 4, 5, 6, 7, 8, 0):  # 找到了
			print(len(L))
			
			return step
		for next in moves[now]:  # 遍历所有可交换位置
			_state = list(state)
			_state[next], _state[now] = _state[now], state[next]  # 交换位置
			_state = tuple(_state)
			if _state not in visited:  # 确认未访问
				q.append((_state, next, step + 1))
		visited.add(state)
	return -1


def pathAnalysis():
	nowlen = len(L[-1])
	for i in range(len(L) - 1, 0, -1):
		print(len(L[i]))
		sum = 0
		for j in range(len(L[i - 1]) - 1):
			if nowlen <= len(moves[L[i - 1][j][1]]) + sum:
				print(L[i][nowlen - 1][0])
				nowlen = j + 1
				break
			else:
				sum += len(moves[L[i - 1][j][1]])


def main():
	res = slidingPuzzle([[2, 8, 5],
	                     [1, 7, 4],
	                     [3, 6, 0]])
	pathAnalysis()
	print(res)


if __name__ == '__main__':
	main()