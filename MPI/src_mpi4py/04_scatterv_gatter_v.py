from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    n_points = 110
    data = list(range(n_points))
    step = int(n_points/size) + 1
    datap = []
    for i in range(0,len(data),step):
        if i+step<len(data):
            datap.append(data[i:i+step])
        else:
            datap.append(data[i:])
    data = datap
else:
    data = None
data = comm.scatter(data, root=0)
comm.Barrier()
print(f'{rank} -- {data}')
data = comm.gather(data, root=0)

if rank == 0:
    datap = []
    for datum in data:
        datap+=datum
    data = datap
    print(f'{rank} -- {data}')
