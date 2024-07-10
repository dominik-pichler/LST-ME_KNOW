```
                                             ______
                                            /      \
                                           |        |
                                           |:/-\\--\.
                                            ( )-( )/,
                                             | ,  .
                                            / \- /. \
                                           | ||L  / \ \
                                          / /  \/    | *
                                         / /          \  \
                                         | |      []   |\ |
                                        /| |           ||  |
                                        || |           ||  |
                                        |  |           ||  |
                                        /_ |__________|||  |
                                       /_ \| ---------||   |
                                       /_ / |         ||   |
                                      /  | ||         | |     
                                      \//  ||         | |  |
                                      /  | ||    T    | |  |
                                     /   | ||    |     |
     "LSTM, Show me the next word"  / 
 ```


# Next Word Predictor
This is a tool to train and evaluate an `LSTM` Model for next word prediction.
For help, just run
```
make help
```

## How to run

### Raw 
**Requirements:**
-  MacOS
- `Python 3.12`

If you want to train the model, just run:  

```
sh run.sh
```

### Docker
For reproducibility, this project is been offered in a containerized version as well. 
In order to run it in a container, all you need to do is run: 



```
docker-compose up 
```
