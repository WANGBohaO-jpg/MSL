class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_list):
        node = self.root
        for token in token_list:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end_of_word = True

    def next_tokens(self, prefix_list):
        node = self.root
        for token in prefix_list:
            if token not in node.children:
                return []
            node = node.children[token]
        return list(node.children.keys())
    
    def valid_tokens(self, token_list):
        valid_tokens_list = [list(self.root.children.keys())]
        node = self.root
        for token in token_list:
            if token in node.children:
                node = node.children[token]
                valid_tokens_list.append(list(node.children.keys()))
            else:
                valid_tokens_list.append([])
        return valid_tokens_list

if __name__ == "__main__":
    # lambda batch_id, sent: trie.next_tokens(sent.tolist()) or [tokenizer.eos_token_id]

    trie = Trie()
    tokens = [[1, 2], [2, 3], [1, 3], [3]]
    for token_list in tokens:
        trie.insert(token_list)

    prefix = [1, 2]
    suffixes = trie.valid_tokens(prefix)
    print(suffixes)
