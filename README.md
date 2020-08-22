## Topic modeling for document sequences

This Java code provides implementations of models for *sequences* of documents, like a thread of forum posts. These are unsupervised models that can discover "topics" as well as how the topics are sequentially related.

It implements the block HMM and the mixed membership Markov model (M4) described in:

Michael J. Paul. [Mixed Membership Markov Models for Unsupervised Conversation Modeling](https://www.aclweb.org/anthology/D12-1009/). Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL 2012). Jeju Island, Korea, July 2012.

### Installation

The code can be compiled with the command:

`javac -cp commons-math-2.1.jar *.java`

### Usage

Run the program with the command:

`java -cp commons-math-2.1.jar:. LearnTopicModel -model <model_name> -input <input_file> -Z <int> [-iters <int>] [<model-specific parameters>]`

- `<model_name>` can be either: `m4` | `hmm`

- `<input_file>` is the filename of the input (format described in a later section).

- The required parameter `-Z` is the number of classes (or topics).

- The optional parameter `-iters` specifies the number of Gibbs sampling iterations to perform. If unspecified, this defaults to 1000.

Additionally, each of the models has its own command-line parameters, which are described below.

When the program finishes, it writes the final variable assignments to the file `<input_file>.assign` -- the output format is similar to the input format, except each word token "word" has been replaced with "word:z:x" where z is the integer class assignment and x is the integer assignment denoting if the token was assigned to the background distribution (0) or not (1).

The variables pi, phi, etc. can be computed from this output. For convenience, python scripts are included to print out the top words for the topics. Each model has its own output script (`topwords_m4.py`, `topwords_hmm.py`), but they are both used in the same way:

`python topwords_m4.py input_docs.txt > output_topwords.txt`

#### M4

The command-line parameters for this model are:

- `-sigma2 <double>`: The value for sigma^2, the variance of the prior over the lambda values. Default 10.0.
- `eta <double>`: The numerator of <eta>/(1000+t) where t is the iteration. This formula gives the gradient ascent step size. Default 1.0.
- `-gamma0 <double>`: The prior weight for belonging to the background. Default 1.0.
- `-gamma1 <double>`: The prior weight for not belonging to the background. Default 1.0.
- `-rightContext <int>`: The interval at which the sampler computes the right context (see below). It can take the following values. -1: Never consider right context (fastest); 0: Always consider right context (correct sampler, but slower); x: Consider right context every x iterations, x > 0. Default 0 (same as 1).

Example usage:

`java -cp commons-math-2.1.jar:. LearnTopicModel -model m4 -input input_docs.txt -iters 5000 -Z 10 -eta 0.1`

##### About the rightContext parameter:

A major source of slowdown in the M4 sampler is the computation of the rightmost term in equation 2 of the paper (the product over the children C). This parameter can tell the sampler to ignore this term (either always or sometimes), by pretending that each block has no children. Without this computation, the sampler becomes about as fast as the standard HMM, but this is an approximation (somewhat related to particle filters or sequential MCMC) and is not expected to learn as good parameters as the correct sampler. I have not done extensive experiments to explore this speed/accuracy tradeoff, but I encourage you to experiment with it, if speed is a concern -- this becomes more of an issue with larger values of Z. 

As an example, supplying this parameter with the value of 5 will tell the sampler to sample from the correct distribution (which includes the right term) every 5 iterations, and to pretend it doesn't exist for the remaining 4/5. This will mean that every fifth iteration will be slower. A value of 0 or 1 will always sample from the correct distribution, and is the default value. 

Note that the current implementation does some caching to avoid naively recomputing the full term for each value of z (by computing what the value would be if no new value is sampled, and only adjusting for the offset induced by the change in z), but it can still be slow.

#### Block HMM 

The command-line parameters for this model are:

- `-gamma0 <double>`: The prior weight for belonging to the background. Default 1.0.
- `-gamma1 <double>`: The prior weight for not belonging to the background. Default 1.0.

Example usage:

`java -cp commons-math-2.1.jar:. LearnTopicModel -model hmm -input input_docs.txt -iters 5000 -Z 10 -gamma0 700.0 -gamma1 300.0`


### Input format

The format of the input file is:

`<doc_id> <doc_parent_id> <doc_string_identifier> <doc_words (space-delimited)>`

Example: 

```0 -1 thread1_message1 this is the beginning of a conversation
1 0 thread1_message2 this is in reply to the first message
2 0 thread1_message3 this is also in reply to the first message
3 2 thread1_message4 this is in reply to the third message
4 -1 thread2_message1 this begins a second conversation
5 -1 thread3_message1 this begins a third conversation```

Each line corresponds to a text block (to use the terminology from the paper). The first column should be an integer ID for the block, and the second column is the integer ID of the block's parent (i.e. the message it is in response to), where the parent ID should correspond to an ID of another block in this input file. The parent ID should be -1 if it has no parent.

The third column is a string ID which could be useful in post-processing. It is required, though you can fill this column with whatever you want. The strings do not need to be unique. 



### Output format

#### M4

The output format is the same as the input format, except each word token "word" has been replaced with "word:z:x" where z is the integer class assignment and x is the integer assignment denoting if the token was assigned to the background distribution (0) or not (1).

The output is written to a file of the same name as the input file, except the filename is appended with ".assign".

Note that the doc IDs and parent IDs (the first two columns of the output) may have changed from the input, based on internal indexing used by the program. The string ID (column 3) will not change.

The learned parameters and hyperparameters are also written to files with extensions ".lambda", ".omega", etc. In particular, the lambda matrix for M4 is written to "input_text.txt.lambda", which is used to show the transitions in the topwords_m4.py script.


#### Block HMM 

The output format is similar to the input format, except each word token "word" has been replaced with "word:x" where x is the integer assignment denoting if the token was assigned to the background distribution (0) or not (1).

The integer class assignment z is printed before the tokens in the line. The  output thus looks like:

`<doc_id> <doc_parent_id> <doc_string_identifier> <z> <doc_words>`

The learned parameters and hyperparameters are also written to files with extensions ".pi", ".omega", etc. In particular, the transition matrix pi is written to "input_text.txt.pi", which is used to show the transitions in the topwords_hmm.py script.


### Viewing the top words

Python scripts are included to print out the top words for the topics. The script 
takes a command line argument of the input file which was used by the Java program. 

Example usage:

`python topwords_m4.py input_docs.txt > output_topwords_m4.txt`

`python topwords_hmm.py input_docs.txt > output_topwords_hmm.txt`



### Troubleshooting

If the program crashes, here are some possible causes:

- There is a blank line in the input (empty document)
- A line does not begin with integers
- A document's parent does not exist (the integer does not match another document in the collection) 



