import random


"""A class represnting a node in an AVL tree"""


class AVLNode(object):
	"""Constructor, you are allowed to add more fields.

	@type value: str
	@param value: data of your node
	"""

	def __init__(self, value=True):
		if type(value) is bool:
			self.value = None
			self.left = None
			self.right = None
			self.parent = None
			self.height = -1  # Balance factor
			self.size = 0
		else:
			virtual = AVLNode()
			self.value = value
			self.left = virtual
			self.right = virtual
			self.parent = None
			self.height = 0  # Balance factor
			self.size = 1

	"""returns the left child
	@rtype: AVLNode
	@returns: the left child of self, None if there is no left child
	"""

	def getLeft(self):
		if self.left != None or self.left.isRealNode():
			return self.left
		return None

	"""returns the right child

	@rtype: AVLNode
	@returns: the right child of self, None if there is no right child
	"""

	def getRight(self):
		if self.right != None or self.right.isRealNode():
			return self.right
		return None

	"""returns the parent 

	@rtype: AVLNode
	@returns: the parent of self, None if there is no parent
	"""

	def getParent(self):
		if self.parent != None:
			return self.parent
		return None

	"""return the value

	@rtype: str
	@returns: the value of self, None if the node is virtual
	"""

	def getValue(self):
		if self != None or self.isRealNode():
			return self.value
		return None

	"""returns the height

	@rtype: int
	@returns: the height of self, -1 if the node is virtual
	"""

	def getHeight(self):
		if self.isRealNode():
			return self.height
		return -1

	"""returns the size

		@rtype: int
		@returns: the size of self, 0 if the node is virtual
		"""

	def getSize(self):
		return self.size

	"""sets left child

	@type node: AVLNode
	@param node: a node
	"""

	def setLeft(self, node):
		self.left = node

	"""sets right child

	@type node: AVLNode
	@param node: a node
	"""

	def setRight(self, node):
		self.right = node

	"""sets parent

	@type node: AVLNode
	@param node: a node
	"""

	def setParent(self, node):
		self.parent = node

	"""sets value

	@type value: str
	@param value: data
	"""

	def setValue(self, value):
		self.value = value

	"""sets the balance factor of the node

	@type h: int
	@param h: the height
	"""

	def setHeight(self, h):
		self.height = h

	"""sets the size of the node

		@type s: int
		@param s: the size
		"""

	def setSize(self, s):
		self.size = s

	"""returns whether self is not a virtual node 

	@rtype: bool
	@returns: False if self is a virtual node, True otherwise.
	"""

	def isRealNode(self):
		return self.height != -1


"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
	"""
	Constructor, you are allowed to add more fields.

	"""

	def __init__(self):
		self.size = 0
		self.root = None
		self.first_item = None
		self.last_item = None

	# add your fields here

	"""returns whether the list is empty

	@rtype: bool
	@returns: True if the list is empty, False otherwise
	"""

	def empty(self):
		return self.size == 0

	"""retrieves the value of the i'th item in the list

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: str
	@returns: the the value of the i'th item in the list
	"""

	def retrieve(self, i):
		if self.empty() or self.length() <= i or i < 0:
			return None
		else:
			node = self.root
			return retrieveNode(node, i, False).value  # retrieveNode is a helping function

	"""inserts val at position i in the list

	@type i: int
	@pre: 0 <= i <= self.length()
	@param i: The intended index in the list to which we insert val
	@type val: str
	@param val: the value we inserts
	@rtype: list
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""

	def insert(self, i, val):

		# Check if tree is empty, if so make node root
		if self.length() == 0:
			node = AVLNode(val)
			self.root = node
			self.size = 1
			self.first_item = node
			self.last_item = node
			return 0

		# Create node from value
		newNode = AVLNode(val)
		newNode.height = 0

		# Insert new node as if BST
		if i != self.length():

			# Retrieve node at index i (successor of new node)
			successor = retrieveNode(self.root, i, True)

			# Find predecessor and insert node after it

			if not successor.left.isRealNode():
				successor.left = newNode
				newNode.parent = successor

			else:
				curr = successor.left
				curr.size += 1
				while curr.right.isRealNode():
					curr = curr.right
					curr.size += 1
				curr.right = newNode
				newNode.parent = curr

		else:  # insert node to right-end of tree
			upToMax = self.root
			upToMax.size += 1
			while upToMax.right.isRealNode():
				upToMax = upToMax.right
				upToMax.size += 1
			upToMax.right = newNode
			newNode.parent = upToMax

		# Updating heights of relevant nodes

		parr = newNode.parent
		while parr is not None:
			if parr.height != 1 + max(parr.left.height, parr.right.height):
				parr.height = 1 + max(parr.left.height, parr.right.height)
				parr = parr.parent
			else:
				break

		# Finding AVL criminals and rebalancing

		parr = newNode.parent
		rebalance_count = 0
		while parr is not None:
			bf = getBF(parr)  # Balance factor
			if (abs(bf) < 2):
				parr = parr.parent
			else:  # |bf| == 2
				if bf == -2:
					bfSon = getBF(parr.right)
					if bfSon == -1:
						leftRotate(parr)
						if self.root == parr:
							self.root = parr.parent
							self.root.parent = None
						rebalance_count = 1
						break
					else:
						rightLeftRotate(parr)
						if self.root == parr:
							self.root = parr.parent
							self.root.parent = None
						rebalance_count = 2
						break

				if bf == 2:
					bfSon = getBF(parr.left)
					if bfSon == -1:
						leftRightRotate(parr)
						if self.root == parr:
							self.root = parr.parent
							self.root.parent = None
						rebalance_count = 2
						break
					else:
						rightRotate(parr)
						if self.root == parr:
							self.root = parr.parent
							self.root.parent = None
						rebalance_count = 1
						break

		# Update first and last item if relevant
		if i == 0:
			first_node = self.root
			while first_node.left.isRealNode():
				first_node = first_node.left
			self.first_item = first_node
		elif i == self.length():
			last_node = self.root
			while last_node.right.isRealNode():
				last_node = last_node.right
			self.last_item = last_node

		# Update tree size
		self.size = self.root.size

		# Update heights again after balancing

		parr = newNode.parent
		while parr is not None:
			if parr.height != 1 + max(parr.left.height, parr.right.height):
				parr.height = 1 + max(parr.left.height, parr.right.height)
			parr = parr.parent

		return rebalance_count

	"""Inserts val at end of list, used during testing
	@type val: str
	@param val: the value we inserts
	@rtype: list
	@returns: the number of rebalancing operation due to AVL rebalancing"""
	def append(self, val):
		return self.insert(self.length(), val)

	"""deletes the i'th item in the list

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: The intended index in the list to be deleted
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""

	def delete(self, i):

		# Return -1 if tree is empty, i is smaller than 0 or i is larger than self.length() - 1
		if self.empty() or i < 0 or i >= self.length():
			return -1

		# Retrieve node at index i
		node = retrieveNode(self.root, i, False)  # the node to delete

		# Deleting node
		# Node is a leaf
		if node.height == 0:
			if node == self.root:  # root is the only node
				self.root = None
				self.size = 0
				self.first_item = None
				self.last_item = None
				return 0
			parr = node.parent
			if parr.right == node:
				parr.right = AVLNode()
				if not parr.left.isRealNode():
					parr.height -= 1
			else:
				parr.left = AVLNode()
				if not parr.right.isRealNode():
					parr.height -= 1

			# Update heights

			while parr is not None:
				parr.size -= 1
				parr.height = 1 + max(parr.left.height, parr.right.height)
				parr = parr.parent
			parr = node.parent


		# Node has one child
		elif node.right.isRealNode() + node.left.isRealNode() == 1:
			if node == self.root:  # root has one child then AVL tree has only two nodes
				if node.right.isRealNode():
					self.root = node.right
					node.right.parent = None
				else:
					self.root = node.left
					node.left.parent = None
				self.size = 1
				self.first_item = self.root
				self.last_item = self.root
				return 0
			parr = node.parent
			if node.right.isRealNode():
				if parr.right == node:
					parr.right = node.right
					node.right.parent = parr
				else:
					parr.left = node.right
					node.right.parent = parr
			if node.left.isRealNode():
				if parr.right == node:
					parr.right = node.left
					node.left.parent = parr
				else:
					parr.left = node.left
					node.left.parent = parr

			# Update heights

			while parr is not None:
				parr.size -= 1
				parr.height = 1 + max(parr.left.height, parr.right.height)
				parr = parr.parent
			parr = node.parent

		# Node has two children

		else:
			successor = node.right
			while successor.left.isRealNode():
				successor = successor.left
			parr = successor.parent

			if successor == node.right:  # successor of node to delete is the first one right
				parr.value = successor.value
				parr.right = successor.right
				parr.right.parent = parr
			elif parr.left == successor:
				parr.left = successor.right
				successor.right.parent = parr
			else:
				parr.right = successor.right
				successor.left.parent = parr
			node.value = successor.value

			# Update heights

			while parr is not None:
				parr.size -= 1
				parr.height = 1 + max(parr.left.height, parr.right.height)
				parr = parr.parent
			parr = successor.parent

		# Finding AVL criminals and rebalancing

		rebalance_count = 0
		parr_update = parr
		while parr is not None:
			bf = getBF(parr)  # Balance factor
			if (abs(bf) < 2):
				parr.height = max(parr.left.height, parr.right.height) + 1
				parr.size = parr.right.size + parr.left.size + 1
				parr = parr.parent
			else:  # |bf| == 2
				if bf == -2:
					bfSon = getBF(parr.right)
					if bfSon == -1 or bfSon == 0:
						leftRotate(parr)
						rebalance_count += 1
						if self.root == parr:
							self.root = parr.parent

					else:
						rightLeftRotate(parr)
						rebalance_count += 2
						if self.root == parr:
							self.root = parr.parent


				if bf == 2:
					bfSon = getBF(parr.left)
					if bfSon == -1:
						leftRightRotate(parr)
						rebalance_count += 2
						if self.root == parr:
							self.root = parr.parent

					else:
						rightRotate(parr)
						rebalance_count += 1
						if self.root == parr:
							self.root = parr.parent
				parr.height = max(parr.left.height, parr.right.height) + 1
				parr.size = parr.right.size + parr.left.size + 1
				parr = parr.parent


		# Update first and last item if relevant
		if i == 0:
			first_node = self.root
			while first_node.left.isRealNode():
				first_node = first_node.left
			self.first_item = first_node
		elif i == self.length() - 1:
			last_node = self.root
			while last_node.right.isRealNode():
				last_node = last_node.right
			self.last_item = last_node


		# Update tree size
		self.size = self.root.size


		return rebalance_count

	"""returns the value of the first item in the list

	@rtype: str
	@returns: the value of the first item, None if the list is empty
	"""

	def first(self):
		if self.first_item is not None:
			return self.first_item.value
		else:
			return None

	"""returns the value of the last item in the list

	@rtype: str
	@returns: the value of the last item, None if the list is empty
	"""

	def last(self):
		if self.last_item is not None:
			return self.last_item.value
		else:
			return None

	"""returns an array representing list 

	@rtype: list
	@returns: a list of strings representing the data structure
	"""

	def listToArray(self):
		array = []
		if self.length() == 0:
			return array
		listToArrayRec(self.root, array)

		return array


	"""returns the size of the list 

	@rtype: int
	@returns: the size of the list
	"""

	def length(self):
		return self.size

	"""sort the info values of the list

	@rtype: list
	@returns: an AVLTreeList where the values are sorted by the info of the original list.
	"""
	def sort(self):
		sortedTree = AVLTreeList()
		array = self.listToArray()
		retArray = merge_sort(array)
		for i in range(len(array)):
			sortedTree.insert(i,retArray[i])

		return sortedTree

	"""permute the info values of the list 

	@rtype: list
	@returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
	"""

	def permutation(self):

		# Copy tree to new tree
		randomTree = AVLTreeList()
		randomTree.root = copy_tree(self.root)
		randomTree.size = self.size

		# Create array from tree and shuffle using Fisher-Yates algorithm
		array = self.listToArray()
		fisherYatesShuffle(array)

		# Copy shuffled array to new tree
		copyArrayToTree(array, randomTree.root, 0)

		# Update first and last items
		first_node = randomTree.root
		while first_node.left.isRealNode():
			first_node = first_node.left
		randomTree.first_item = first_node

		last_node = randomTree.root
		while last_node.right.isRealNode():
			last_node = last_node.right
		randomTree.last_item = last_node

		return randomTree

	"""concatenates lst to self

	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@rtype: int
	@returns: the absolute value of the difference between the height of the AVL trees joined
	"""

	def concat(self, lst):

		# Taking care of edge cases
		if self.length() == 0 and lst.length() == 0:
			return 0
		elif self.length() == 0:
			self.root = lst.root
			self.size = lst.size
			self.first_item = lst.first_item
			self.last_item = lst.last_item
			return self.root.height
		elif lst.length() == 0:
			return self.root.height

		# Storing absolute value of the difference between the heights
		abs_height = abs(lst.root.height - self.root.height)

		# Finding which tree is higher and acting accordingly
		if lst.root.height >= self.root.height:
			concat_tree = lst

			# Storing and deleting last node in first tree
			connect = AVLNode(self.last())
			self.delete(self.length() - 1)

			# Finding first node in second tree with height that is taller by one than root of first tree
			replace = lst.root
			if self.root is not None:
				while replace.height > self.root.height + 1:
					replace = replace.left
			else:
				replace = lst.first_item

			# Replacing said node with last node in first tree and changing its left to the first tree and right to node
			new_replace = AVLNode(replace.value)
			new_replace.right = replace.right
			new_replace.right.parent = new_replace
			new_replace.left = replace.left
			new_replace.left.parent = new_replace
			new_replace.size = replace.size
			new_replace.height = replace.height

			replace.value = connect.value
			if self.root is not None:
				replace.left = self.root
				self.root.parent = replace
			else:
				replace.left = AVLNode()
			replace.right = new_replace
			new_replace.parent = replace
			replace.size = replace.left.size + replace.right.size + 1
			replace.height = max(replace.left.height, replace.right.height) + 1

		else:
			concat_tree = self

			# Storing and deleting first node in second tree
			connect = AVLNode(lst.first())
			lst.delete(0)

			# Finding first node in first tree with height is taller by one than root of second tree
			replace = self.root

			if lst.root is not None:
				while replace.height > lst.root.height + 1:
					replace = replace.right
			else:
				replace = self.last_item

			# Replacing said node with first node in second tree and changing its right to the second tree and left to node
			new_replace = AVLNode(replace.value)
			new_replace.right = replace.right
			new_replace.right.parent = new_replace
			new_replace.left = replace.left
			new_replace.left.parent = new_replace
			new_replace.size = replace.size
			new_replace.height = replace.height

			replace.value = connect.value
			if lst.root is not None:
				replace.right = lst.root
				lst.root.parent = replace
			else:
				replace.right = AVLNode()
			replace.left = new_replace
			new_replace.parent = replace
			replace.size = replace.left.size + replace.right.size + 1
			replace.height = max(replace.left.height, replace.right.height) + 1

		# Storing suspect node where rebalancing checks start
		sus_node = replace

		# Fixing heights of parents before rebalancing begins
		parr = sus_node.parent

		while parr is not None:
			parr.height = max(parr.left.height, parr.right.height) + 1
			parr = parr.parent

		# Rebalancing as if delete
		parr = sus_node.parent
		while parr is not None:
			bf = getBF(parr)  # Balance factor
			if (abs(bf) < 2):
				parr.height = max(parr.left.height, parr.right.height) + 1
				parr.size = parr.right.size + parr.left.size + 1
				parr = parr.parent
			else:  # |bf| == 2
				if bf == -2:
					bfSon = getBF(parr.right)
					if bfSon == -1 or bfSon == 0:
						leftRotate(parr)
						if concat_tree.root == parr:
							concat_tree.root = parr.parent

					else:
						rightLeftRotate(parr)
						if concat_tree.root == parr:
							concat_tree.root = parr.parent


				if bf == 2:
					bfSon = getBF(parr.left)
					if bfSon == -1:
						leftRightRotate(parr)
						if concat_tree.root == parr:
							concat_tree.root = parr.parent

					else:
						rightRotate(parr)
						if concat_tree.root == parr:
							concat_tree.root = parr.parent
				parr.height = max(parr.left.height, parr.right.height) + 1
				parr.size = parr.right.size + parr.left.size + 1
				parr = parr.parent

		# Update first and last items of concat_tree
		first_node = concat_tree.root
		while first_node.left.isRealNode():
			first_node = first_node.left
		concat_tree.first_item = first_node

		last_node = concat_tree.root
		while last_node.right.isRealNode():
			last_node = last_node.right
		concat_tree.last_item = last_node

		# Updating concat tree size
		concat_tree.size = concat_tree.root.size

		# Updating self with concat_tree
		self.root = concat_tree.root
		self.size = concat_tree.size
		self.first_item = concat_tree.first_item
		self.last_item = concat_tree.last_item

		return abs_height

	"""searches for a *value* in the list

	@type val: str
	@param val: a value to be searched
	@rtype: int
	@returns: the first index that contains val, -1 if not found.
	"""

	def search(self, val):
		array = self.listToArray()
		for i in range(len(array)):
			if array[i] == val:
				return i

		return -1

	"""returns the root of the tree representing the list

	@rtype: AVLNode
	@returns: the root, None if the list is empty
	"""

	def getRoot(self):
		return self.root


"""
retrieves node using Tree-Select, as seen in class
If resize is True, the function enlarges size of node it passes by one. This is used in insert
"""
# Used in retrieve, insert and delete
def retrieveNode(node, i, resize):

	while True:
		if resize is False:
			r = node.left.size
			if r == i:
				return node
			elif r > i:
				node = node.left
			else:
				node = node.right
				i = i - r - 1
		else:
			node.size += 1
			r = node.left.size
			if r == i:
				return node
			elif r > i:
				node = node.left
			else:
				node = node.right
				i = i - r - 1

# Used in insert, delete and concat
def leftRotate(node):
	bf = getBF(node)
	tmp = AVLNode(node.right.value)
	tmp.left = node
	tmp.parent = node.parent
	tmp.right = node.right.right
	if node.right.right.isRealNode():
		node.right.right.parent = tmp
	tmp.height = node.right.height
	if node.right.left.isRealNode():
		node.right.left.parent = node
		node.right = node.right.left
	else:
		node.right = AVLNode()

	if node.parent is not None:
		if node.parent.right.value == node.value:
			node.parent.right = tmp
		else:
			node.parent.left = tmp
	node.parent = tmp

	#Update sizes of nodes
	node.size = node.right.size + node.left.size + 1
	tmp.size = tmp.right.size + tmp.left.size + 1

	if bf == -2:
		node.height -= 2
	elif bf == -1:
		node.height -= 1
		tmp.height += 1

# Used in insert, delete and concat
def rightRotate(node):
	bf = getBF(node)
	tmp = AVLNode(node.left.value)
	tmp.right = node
	tmp.parent = node.parent
	tmp.left = node.left.left
	if node.left.left.isRealNode():
		node.left.left.parent = tmp
	tmp.height = node.left.height
	if node.left.right.isRealNode():
		node.left.right.parent = node
		node.left = node.left.right
	else:
		node.left = AVLNode()

	if node.parent is not None:
		if node.parent.right.value == node.value:
			node.parent.right = tmp
		else:
			node.parent.left = tmp
	node.parent = tmp

	# Update sizes of nodes
	node.size = node.right.size + node.left.size + 1
	tmp.size = tmp.right.size + tmp.left.size + 1


	if bf == 2:
		node.height -= 2
	elif bf == 1:
		node.height -= 1
		tmp.height += 1

# Used in insert, delete and concat
def leftRightRotate(node):
	leftRotate(node.left)
	rightRotate(node)

# Used in insert, delete and concat
def rightLeftRotate(node):
	rightRotate(node.right)
	leftRotate(node)

# Used in insert, delete and concat
def getBF(node):
	return node.left.height - node.right.height

# Used in listToArray
def listToArrayRec(node, array):
	if node.height == 0:
		array.append(node.value)
	else:
		if node.left.isRealNode():
			listToArrayRec(node.left, array)
		array.append(node.value)
		if node.right.isRealNode():
			listToArrayRec(node.right, array)

# Used in permutation
def copy_tree(root):
	if not root.isRealNode():
		return AVLNode()

	new_root = AVLNode(root.value)
	new_root.size = root.size
	new_root.height = root.height
	new_root.right = copy_tree(root.right)
	new_root.left = copy_tree(root.left)

	new_root.left.parent = new_root
	new_root.right.parent = new_root

	return new_root

# Used in permutation
def fisherYatesShuffle(array):
	for i in range(len(array) - 1, 0, -1):
		j = random.randint(0, i)
		tmp = array[i]
		array[i] = array[j]
		array[j] = tmp

# Used in permutation
def copyArrayToTree(array, node, i):
	if not node.right.isRealNode() and not node.left.isRealNode():
		node.value = array[i]
		return i + 1
	if node.left.isRealNode():
		i = copyArrayToTree(array, node.left, i)
	node.value = array[i]
	if node.right.isRealNode():
		i = copyArrayToTree(array, node.right, i + 1)
		return i

	return i + 1

# Used in merge_sort
def merge(left,right):
	ret=[]
	Lindex=0
	Rindex=0
	while Lindex<len(left) and Rindex<len(right):
		if left[Lindex] is None:
			ret.append(left[Lindex])
			Lindex += 1
		elif right[Rindex] is None:
			ret.append(right[Rindex])
			Rindex += 1
		elif(left[Lindex]<right[Rindex]):
			ret.append(left[Lindex])
			Lindex+=1
		else:
			ret.append(right[Rindex])
			Rindex+=1

	ret.extend(left[Lindex:])
	ret.extend(right[Rindex:])
	return ret

# Used in sort
def merge_sort(arr):
	if len(arr) <= 1:
		return arr

	middle = len(arr) // 2
	left = arr[:middle]
	right = arr[middle:]

	left = merge_sort(left)
	right = merge_sort(right)

	return merge(left, right)
