class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def find_common_ancestor(root, node1, node2):
    if root is None:
        return None

    # If either of the nodes is the root, then root is the common ancestor
    if root == node1 or root == node2:
        return root

    # Recursively search for the nodes in the left and right subtrees
    left_ancestor = find_common_ancestor(root.left, node1, node2)
    right_ancestor = find_common_ancestor(root.right, node1, node2)

    # If both nodes are found in different subtrees, then the current root is the common ancestor
    if left_ancestor and right_ancestor:
        return root

    # If one node is found, return it; otherwise, return the other
    return left_ancestor if left_ancestor else right_ancestor

# Example usage:
# Construct a sample binary tree
#        3
#       / \
#      5   1
#     / \ / \
#    6  2 0  8
#      / \
#     7   4

root = TreeNode(3)
root.left = TreeNode(5)
root.right = TreeNode(1)
root.left.left = TreeNode(6)
root.left.right = TreeNode(2)
root.right.left = TreeNode(0)
root.right.right = TreeNode(8)
root.left.right.left = TreeNode(7)
root.left.right.right = TreeNode(4)

# Find the first common ancestor of nodes with values 5 and 1
ancestor = find_common_ancestor(root, root.left.left, root.left.right)
print("First common ancestor:", ancestor.value)
