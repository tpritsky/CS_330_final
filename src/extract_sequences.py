import json

#load input data json
f = open('data/protein_tasks_dict.json')
data = json.load(f)
prot_seqs = list(data.keys())

#output to fasta file
f = open('data/prot_seqs.fasta', 'w+')
for i in range(len(prot_seqs)):
    f.write('>protein_' + str(i)+'\n' + str(prot_seqs[i]))
    if i!=len(prot_seqs)-1:
        f.write('\n')

f.close()