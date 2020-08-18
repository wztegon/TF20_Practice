L = []
moves = [(1, 3),    (0, 2, 4),    (1, 5),
         (0, 4, 6), (1, 3, 5, 7), (2, 4, 8),
         (3, 7),    (4, 6, 8),    (5, 7)]  # 每个位置的0可以交换的位置


def slidingPuzzle(board):
	board = board[0] + board[1] + board[2] # 把board连起来变一维
	
	q= [((tuple(board), board.index(0), 0), (tuple(board), board.index(0), 0))] # bfs队列和已访问状态记录
	visited = set()
	while q:
		
		(cur, pre) = q.pop(0)  # 分别代表当前状态，0的当前位置和当前步数
		L.append((cur, pre))
		state, now, step = cur

		
		if state == (1, 2, 3, 4, 5, 6, 7, 8, 0):  # 找到了
			print(len(L))
			return step
		
		for next in moves[now]:  # 遍历所有可交换位置
			_state = list(state)
			_state[next], _state[now] = _state[now], state[next]  # 交换位置
			_state = tuple(_state)
			if _state not in visited:  # 确认未访问
				q.append(((_state, next, step + 1), (state, now, step)))
		visited.add(state)
	return -1


def pathAnalysis(res):
	print(L[-1][0])
	target = L[-1]
	cur_index = len(L) - 1
	for i in range(res):
		for index in range(cur_index, 0, -1):
			#print(item)
			if L[index][0] == target[1]:
				print(L[index][0])
				target = L[index]
				cur_index = index
				break
	print(L[0][0])
def main():
	# res = slidingPuzzle([[2, 8, 5],
	#                      [1, 7, 4],
	#                      [3, 6, 0]])
	res = slidingPuzzle([[4, 1, 0],
	                     [7, 6, 3],
	                     [5, 2, 8]])
	pathAnalysis(res)
	print(res)


if __name__ == '__main__':
	main()