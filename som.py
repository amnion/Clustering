
def SOM(freqBand,nIterations,gamma,sigma=1,nNodes=10,plot=False):
    
    # I make the data 2-d, adding the position as an explicit column
    X = np.hstack((np.arange(freqBand.shape[1]).reshape(-1,1),freqBand.T))
    
    # Initialize the SOM with random weights
    #Som = np.random.randint(max(freqBand[0]), size=(nNodes,2))
    mapFrq = np.random.randint(max(freqBand[0]), size=(nNodes,1))
    mapPos = np.random.randint(freqBand.shape[1], size=(nNodes,1))
    Som = np.hstack((mapPos,mapFrq))
    
    # Initial params
    sigma = round(nNodes/5)
    gamma = gamma #/nIterations
    lamb = nIterations*100  # Bigger lambda, slower learning and influence area decay
    
    if plot:
        fig = plt.figure(figsize=(18,6))
    for i in range(nIterations):
        sample = X[np.random.choice(X.shape[0])] # Pick a random sequence chunk from X
        
        def drawit():
            plt.clf()
            #plt.plot(Som[:,1])
            plt.scatter(X[:,0],X[:,1],s=80,c="blue",alpha=0.1)
            plt.scatter(Som[:,0],Som[:,1],s=70,marker="s",c="red",alpha=0.5)
            plt.scatter(sample[0],sample[1],s=100,marker="s",c="lime")
            plt.xlim([0,X.shape[0]])
            ipd.clear_output(wait=True)
            ipd.display(fig)
            #time.sleep(0.0001)
        if plot:
            drawit()

        # Find the node in the SOM that best matches the input data
        distances = np.zeros((nNodes,1))
        for k in range(nNodes):
            distances[k] = np.sqrt(sum((sample-Som[k])**2)) 
        bestNode = np.argmin(distances)

        # Find the neighborhood of the bestNode - I'll represent linearly for now
        sigma = int(round( sigma * np.exp(-i/lamb) ))
        gamma = gamma * np.exp(-i/lamb)
        for j in range((bestNode-sigma),(bestNode+sigma)):
            if j > 0 and j < nNodes:
                dist = float(abs(j-bestNode))
                phi = np.exp( -(dist**2) / (2*(float(sigma)**2)) )
                Som[j] = Som[j] + gamma*phi*sigma*(sample-Som[j])
             
    if plot:
        drawit()
        ipd.clear_output(wait=True)
                
    print('SOM trained {} nodes for {} iterations.'.format(nNodes,nIterations))
    return Som
