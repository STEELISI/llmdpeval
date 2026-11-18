from mergexp import *

# One CPU node for evaluating generated code and running the paper apps (no GPU).
net = Network('dpeval', addressing==ipv4, routing==static)
eval_node = net.node('eval', proc.cores>=16, memory.capacity>=gb(64), disk.capacity==gb(200))
experiment(net)
