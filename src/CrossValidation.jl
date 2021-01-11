module CrossValidation
	
	include("./MatrixNetworks.jl")
	include("./NetworkBasedInference.jl")
	using Random
	using LinearAlgebra
	using .MatrixNetworks
	using .NetworkBasedInference

	export generateFolds, split_adjmat, alaimo2013_split_adjmat, cheng2012_split_adjmat
	export SDTNBI_LOOCV

	"""
		_10_fold_over_edges(A::Matrix, rng::Number = rand(1:typemax(Int64)))::Dict

	Generate 10 groups of edges for cross-validation.
	
	# Examples
	```julia-repl
	julia> A = [ 0 0 1 0 1 ; 0 0 0 0 1 ; 1 0 0 0 0 ; 0 0 0 0 0 ; 1 1 0 0 0 ]
	5×5 Array{Int64,2}:
	0  0  1  0  1
	0  0  0  0  1
	1  0  0  0  0
	0  0  0  0  0
	1  1  0  0  0
	julia> generateFolds_edges_10fold(A)
	Dict{Tuple{Int64,Int64},Int64} with 25 entries:
	  (3, 4) => 3
  	  (4, 3) => 3
  	  (2, 3) => 1
  	  (4, 4) => 7
  	  (2, 2) => 9
  	  (5, 4) => 6
  	  (2, 1) => 9
  	  (2, 5) => 2
  	  (3, 1) => 8
  	  (1, 4) => 2
  	  (3, 3) => 2
  	  (4, 5) => 0
  	  (1, 3) => 1
  	  (3, 2) => 3
  	  (5, 2) => 0
  	  (2, 4) => 5
  	  (1, 1) => 4
  	  (1, 2) => 5
  	  (5, 1) => 4
  	  (5, 5) => 5
  	  (5, 3) => 7
  	  (4, 1) => 4
  	  (1, 5) => 6
  	  (3, 5) => 1
  	  (4, 2) => 8
	```

	# Arguments
	- `M::Matrix`: rectangular M × N adjacency matrix.
	- `rng::Int`: seed for random number generator.

	"""
	function _10_fold_over_edges(A::Matrix, rng::Number = rand(1:typemax(Int64)))::Dict
		# Get all possible edges in graph
		M, N		= size(A)
		all_edges	= [(i,j) for i in 1:M for j in 1:N]

		# Assign fold randomly to all possible edges in graph
		seed			= Random.seed!(rng)
		shuffled_edges	= shuffle(seed, all_edges)
		edges_fold		= Dict((i,j) => mod(f,10) for (f,(i,j)) 
							   in enumerate(shuffled_edges))

		return edges_fold
	end
	
	"""
		generateFolds_edges_loo(A::Matrix, rng::Number = rand())::Dict

	Generate leave-one-out groups of edges for cross-validation.
	
	# Examples
	```julia-repl
	julia> A = [ 0 0 1 0 1 ; 0 0 0 0 1 ; 1 0 0 0 0 ; 0 0 0 0 0 ; 1 1 0 0 0 ]
	5×5 Array{Int64,2}:
	0  0  1  0  1
	0  0  0  0  1
	1  0  0  0  0
	0  0  0  0  0
	1  1  0  0  0
	julia> generateFolds_edges_loo(A, randint())
	Dict{Tuple{Int64,Int64},Int64} with 25 entries:
	  (3, 4) => 24
	  (4, 3) => 20
	  (2, 3) => 16
	  (2, 2) => 15
	  (4, 4) => 19
	  (5, 4) => 2
	  (2, 1) => 6
	  (2, 5) => 13
	  (3, 1) => 4
	  (1, 4) => 22
	  (3, 3) => 1
	  (4, 5) => 21
	  (1, 3) => 23
	  (5, 2) => 17
	  (3, 2) => 25
	  (2, 4) => 7
	  (1, 1) => 8
	  (1, 2) => 9
	  (5, 1) => 5
	  (5, 3) => 10
	  (4, 1) => 11
	  (5, 5) => 14
	  (1, 5) => 3
	  (4, 2) => 12
	  (3, 5) => 18
	```

	# Arguments
	- `M::Matrix`: rectangular M × N adjacency matrix.
	- `rng::Int`: seed for random number generator.

	"""
	function _leave_one_out_over_edges(A::Matrix, rng::Number = rand(1:typemax(Int64)))::Dict
		# Get all possible edges in graph
		M, N		= size(A)
		all_edges	= [(i,j) for i in 1:M for j in 1:N]

		# Assign fold randomly to all possible edges in graph
		seed			= Random.seed!(rng)
		shuffled_edges	= shuffle(seed, all_edges)
		edges_fold		= Dict((i,j) => f for (f,(i,j)) in enumerate(shuffled_edges))

		return edges_fold
	end
	
	"""
		generateFolds_nodes_loo(A::Matrix, rng::Number = rand())::Dict

	Generate leave-one-out groups of edges for cross-validation.
	
	# Examples
	```julia-repl
	julia> A = [ 0 0 1 0 1 ; 0 0 0 0 1 ; 1 0 0 0 0 ; 0 0 0 0 0 ; 1 1 0 0 0 ]
	5×5 Array{Int64,2}:
	0  0  1  0  1
	0  0  0  0  1
	1  0  0  0  0
	0  0  0  0  0
	1  1  0  0  0
	```

	# Arguments
	- `M::Matrix`: rectangular M × N adjacency matrix.
	- `rng::Int`: seed for random number generator.

	"""
	function _10_fold_over_nodes(A::Matrix, rng::Number = rand(1:typemax(Int64)))::Dict
		# Get all possible edges in graph
		M, N		= size(A)
		all_edges	= [(i,j) for i in 1:M for j in 1:N]

		# Assign fold randomly to all possible edges in graph
		seed			= Random.seed!(rng)
		shuffled_edges	= shuffle(seed, all_edges)
		edges_fold		= Dict((i,j) => f for (f,(i,j)) in enumerate(shuffled_edges))

		return edges_fold
	end
	
	function _leave_one_out_over_nodes(A::Matrix, rng::Number)::Dict
		# Get all possible edges in graph
		M, N		= size(A)
		all_edges	= [(i,j) for i in 1:M for j in 1:N]

		# Assign fold randomly to all possible edges in graph
		seed			= Random.seed!(rng)
		shuffled_edges	= shuffle(seed, all_edges)
		edges_fold		= Dict((i,j) => f for (f,(i,j)) in enumerate(shuffled_edges))

		return edges_fold
	end
	
	"""
		generateFolds(A::Matrix, method::String = "10-fold over edges")::Dict
	
	Wrapper function for generating fold groups
	"""
	function generateFolds(A::Matrix, method::String = "10-fold over edges", rng::Number = rand(1:typemax(Int64)))::Dict
		if method == "10-fold over edges"
			return _10_fold_over_edges(A, rng)
		elseif method == "Leave-one-out over edges"
			return _leave_one_out_over_edges(A, rng)
		elseif method == "10-fold over nodes"
			return _10_fold_over_nodes(A, rng)
		elseif method == "Leave-one-out over nodes"
			return _leave_one_out_over_nodes(A, rng)
		else
			error("$method not available. Available methods for fold generation: 
				  - 10-fold over edges
				  - 10-fold over nodes
				  - Leave-one-out over edges
				  - Leave-one-out over nodes")
		end
	end

	"""

	"""
	function split_adjmat(A::Matrix, groups::Dict, fold::Int)
		# Create empty matrices
		M, N = size(A)
		training_A = deepcopy(A)
		test_A = zeros(Int64, M, N)

		# Get edges for given group
		test_edges = [e for (e,f) in groups if f == fold]
		
		# Populate matrices
		for (i,j) in test_edges
			training_A[i,j] = 0
			test_A[i,j] = A[i,j]
		end

		return training_A, test_edges
	end

	"""
		alaimo2013_split_adjmat(A::Matrix, groups::Dict, fold::Int)

	Generate test and training sets for cross-validation based on Alaimo et al. (2013)

	"""
	#=
	function alaimo2013_split_adjmat(A::Matrix, groups::Dict, fold::Int)
		# Create empty matrices
		M, N = size(A)
		training_A = deepcopy(A)
		test_A = zeros(Int64, M, N)

		# Get edges for given group
		test_edges = [e for (e,f) in groups if f == fold]
		
		# Populate matrices
		for (i,j) in test_edges
			j_adjacency = deepcopy(training_A[:,j])
			i_adjacency = deepcopy(training_A[i,:])
			j_adjacency[i] = 0
			i_adjacency[j] = 0

			if sum(j_adjacency) == 0 || sum(i_adjacency) == 0
				continue
				#filter!(edge->edge≠(i,j),test_edges)
			else
				training_A[i,j] = 0
				test_A[i,j] = A[i,j]
			end
		end

		return training_A, test_A, test_edges
	=#	
	"""
	cheng2012_split_adjmat(A::Matrix, groups::Dict, fold::Int)

	Generate test and training sets for cross-validation based on Cheng et al. (2012)

	"""
	function cheng2012_split_adjmat(A::Matrix, groups::Dict, fold::Int)
		# Create empty matrices
		M, N		= size(A)
		training_A	= deepcopy(A)
		test_A		= zeros(Int64, M, N)

		# Get edges for given group
		test_edges_0 = [e for (e,f) in groups if f == fold]
		test_edges_1 = []
		
		# Populate matrices
		for (i,j) in test_edges_0
			training_A[i,j] = 0
			test_A[i,j] = A[i,j]
		end

		# Check if any vertex is isolated
		for i in 1:1:M
			if sum(training_A[i,:]) != 0
				edges_to_keep = [(i1, j1) for (i1, j1) in test_edges_0 if i1 == i]
				for e in edges_to_keep
					push!(test_edges_1, e)
				end
			end
		end

		for j in 1:1:N
			if sum(training_A[:,j]) != 0
				edges_to_keep = [(i1, j1) for (i1, j1) in test_edges_0 if j1 == j]
				for e in edges_to_keep
					push!(test_edges_1, e)
				end
			end
		end

		return training_A, test_A, test_edges_0, Set(test_edges_1)
	end
	
	function alaimo2013_split_adjmat(A::Matrix, groups::Dict, fold::Int)
		# Create empty matrices
		M, N		= size(A)
		training_A	= deepcopy(A)
		test_A		= zeros(Int64, M, N)

		# Get edges for given group
		test_edges = [e for (e,f) in groups if f == fold]
		
		# Populate matrices
		for (i,j) in test_edges
			training_A[i,j] = 0
			test_A[i,j] = A[i,j]
		end

		# Check if any vertex is isolated
		for i in 1:1:M
			if sum(training_A[i,:]) == 0
				i_neighbours = neighbours(A, i, "row")
				training_A[i, rand(i_neighbours)] = 1
			end
		end

		for j in 1:1:N
			if sum(training_A[:,j]) == 0
				j_neighbours = neighbours(A, j, "column")
				training_A[rand(j_neighbours), j] = 1
			end
		end

		return training_A, test_A, test_edges
	end

	function SDTNBI_LOOCV(DS::Matrix, DT::Matrix, C::Int, k::Int = 2)
		# Get number of nodes per type from adjacency matrices
		Nc, Nd, Ns, Nt = (1, size(DS, 1), size(DS, 2), size(DT, 2))
		
		# Create matrix A
		Mcc = zeros(Nc, Nc)
		Mcd = zeros(1, Nd-1)
		Mcs = DS[C,:]'
		Mct = zeros(1, Nt)
			
		Mdc = Mcd'
		Mdd = zeros(Nd-1, Nd-1)
		Mds = DS[1:end .!= C,:]
		Mdt = DT[1:end .!= C,:]
			
		Msc = Mcs'
		Msd = Mds'
		Mss = zeros(Ns, Ns)
		Mst = zeros(Ns, Nt)
			
		Mtc = Mct'
		Mtd = Mdt'
		Mts = Mst'
		Mtt = zeros(Nt, Nt)
					
		A = [ Mcc Mcd Mcs Mct ; 
			  Mdc Mdd Mds Mdt ; 
			  Msc Msd Mss Mst ;
			  Mtc Mtd Mts Mtt ]
			
		# Create matrix B
		B = deepcopy(A)
		B[1,:] = B[1,:] .= 0
		B[:,1] = B[:,1] .= 0
			
		# Calculate transfer matrix W
		W = SDTNBI(Symmetric(B))
		F = A * W^k
		R = F[1, Ns+Nd+1:end]
		
		return R
	end

end
