Mr.LDA
=======

Mr.LDA is an open-source package for flexible, scalable, multilingual topic
modeling using variational inference in MapReduce. For more details, please consult [this paper](http://www2012.org/proceedings/proceedings/p879.pdf). The latest version of the code can always be found on [GitHub](https://github.com/lintool/Mr.LDA).

Please send any bugs reports or questions to Ke Zhai (kzhai@umd.edu).

Getting Started
---------------

Clone the repo:

```
$ git clone git@github.com:lintool/Mr.LDA.git
```

Then build using the standard invocation:

```
$ mvn clean package
```

If you want to set up your Eclipse environment:

```
$ mvn eclipse:clean
$ mvn eclipse:eclipse
```

Corpus Preparation
------------------

Some sample data from the Associated Press can be found in this [separate repo](https://github.com/lintool
/Mr.LDA-data). This is the same sample data that is used in [Blei's LDA implementation in C](http://www.cs.princeton.edu/~blei/lda-c/).

The repo includes a Python script for parsing the corpus into a format that Mr.LDA uses. The output of the script is stored in `ap-sample.txt.gz`. This is the data file that you'll want to load in HDFS.

Mr.LDA takes plain text files as input, where each line in the text file represents a document. The document id and content are separated by a *tab*, and words in the content are separated by a spaces. For example, the first two lines of `ap-sample.txt` look like:

```
ap881218-0003   student privat baptist school allegedli kill ...
ap880224-0195   bechtel group offer sell oil israel discount ...
```

To prepare the corpus into the internal format used by Mr.LDA, run the following command:

```
$ hadoop jar target/mrlda-0.9.0-SNAPSHOT-fatjar.jar cc.mrlda.ParseCorpus \
    -input ap-sample.txt -output ap-sample-parsed
```

When you examine the output, you'll see:

```
$ hadoop fs -ls ap-sample-parsed
ap-sample-parsed/document
ap-sample-parsed/term
ap-sample-parsed/title
```

The directory `term` stores the mapping between a unique token and its unique integer id used internally (i.e., the dictionary). The directory `title` stores the mapping between the document id and its unique integer internal id. These are both stored in `SequenceFiles` format, with `IntWritable` as the key and `Text` as the value.

To example the first 20 document id mappings:

```
$ hadoop jar target/mrlda-0.9.0-SNAPSHOT-fatjar.jar \
    edu.umd.cloud9.io.ReadSequenceFile ap-sample-parsed/title 20
```

And to example the first 20 terms of the dictionary:

```
$ hadoop jar target/mrlda-0.9.0-SNAPSHOT-fatjar.jar \
    edu.umd.cloud9.io.ReadSequenceFile ap-sample-parsed/term 20
```

Running "Vanilla" LDA
---------------------

Mr.LDA implements LDA using variational inference. Here's an invocation for running 50 iterations on the sample dataset:

```
$ nohup hadoop jar target/mrlda-0.9.0-SNAPSHOT-fatjar.jar \
    cc.mrlda.VariationalInference \
    -input ap-sample-parsed/document -output ap-sample-lda \
    -term 10000 -topic 20 -iteration 50 -mapper 50 -reducer 20 >& lda.log &
```

The above command will put the process in the background and you can `tail -f lda.log` to see its process.

Note that `-term` option specifies the number of unique tokens in the corpus. This just needs to be a reasonable upper bound.

If the MapReduce jobs are interrupted for any reason, you can restart at a particular iteration with the `-modelindex` parameter. For example, to pick up where the previous command left off and run another 10 iterations, do this:

```
$ nohup hadoop jar target/mrlda-0.9.0-SNAPSHOT-fatjar.jar \
    cc.mrlda.VariationalInference \
    -input ap-sample-parsed/document -output ap-sample-lda \
    -term 10000 -topic 20 -iteration 60 -mapper 50 -reducer 20 \
    -modelindex 50 >& lda.log &
```

Evaluation on 20newsgroup
-------------------------

In this section, we will evaluate Mr. LDA using held-out likelihood and topic coherence on the 20newsgroup data.
The key to evaluation of any machine learning algorithm is to split the corpus into three datasets: *training set*, *development set*, and *test set*. The *training set* is used to fit the model, the *development set* is used to select parameters, and the *test set* is used for evaluation. For this task, since we do not focus on tuning parameters, we use only the training set and test set.

**Step 1: Preprocessing**

The sample split and scripts for preprocessing 20newsgroup can be found in [separate repo](https://github.com/lintool/Mr.LDA-data). Clone the repo and use it as a working directory:

```
$ git clone git@github.com:lintool/Mr.LDA-data.git
$ cd Mr.LDA-data
```

Download the 20newsgroup data from [here](http://qwone.com/~jason/20Newsgroups/):

```
$ wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
$ tar -xzvf 20news-bydate.tar.gz
```

Unpack the following file:

```
$ tar xvfz 20news-labels.tgz
```

Preprocess the collection:

```
# 50/500 are min/max document frequencies for a word, alter these numbers as you wish
$ python parse_20news/createLdacMrlda.py 20news-labels/train.labels \
   20news-labels/dev.labels 20news-labels/test.labels 50 500
```

The above command generates the following files:
 
+ LDAC files: `20news.ldac.train`, `20news.ldac.dev`, `20news.ldac.test`
+ Mr. LDA files: `20news.mrlda.train`, `20news.mrlda.dev`, `20news.mrlda.test`
+ Raw data files: `20news.raw.train`, `20news.raw.dev`, `20news.raw.test`
+ Vocabulary file: `20news.vocab.txt`
+ Statistics and final labels: `20news.stat.train`, `20news.stat.dev`, `20news.stat.test`

**Step 2: Run Mr. LDA**

Follow the below steps to run Mr. LDA:

```
# Copy data to hdfs
$ hadoop fs -put 20news.mrlda.train

# Parse corpus
$ hadoop jar target/mrlda-0.9.0-SNAPSHOT-fatjar.jar cc.mrlda.ParseCorpus \
   -input 20news.mrlda.train -output 20news.mrlda.train-parsed

# Run Vanilla LDA with symmetric alpha
$ nohup hadoop jar target/mrlda-0.9.0-SNAPSHOT-fatjar.jar \
    cc.mrlda.VariationalInference \
    -input 20news.mrlda.train-parsed/document \
    -output 20news.mrlda.train-lda \
    -symmetricalpha 0.01 \
    -topic 20 -term 100000 -iteration 1000 \
    -mapper 50 -reducer 20 >& 20news.log &
```

**Step 3: Compute held-out likelihood**

We use [Blei's LDA implementation in C](http://www.cs.princeton.edu/~blei/lda-c/) (LDAC) to compute held-out likelihood. LDAC requires a `.beta` file and `.other` file to compute held-out score on unseen data.

First, download and untar LDAC:

```
$ wget http://www.cs.princeton.edu/~blei/lda-c/lda-c-dist.tgz
$ tar -xzvf lda-c-dist.tgz
$ cd lda-c-dist
$ make
```

Grab the `beta` file from Mr. LDA and convert it to LDAC format:

```
# Grab beta file from hdfs, ITERATION is the iteration where Mr. LDA converges
$ hadoop jar target/mrlda-0.9.0-SNAPSHOT-fatjar.jar \
    edu.umd.cloud9.io.ReadSequenceFile \
    20news.mrlda.train-lda/beta-ITERATION > 20news.mrlda.train.20.beta

# convert to proper format for LDAC
$ python parse_20news/convertMrldaBetaToBeta.py 20news.mrlda.train.20.beta \
    20news.mrlda.train.20.ldac.beta VOCAB_SIZE
```

While `VOCAB_SIZE` is the size of vocabulary, you can get this number by `wc 20news.vocab.txt`.

Create `.other` file and name it `20news.mrlda.train.20.ldac.other`. This file contains values for `alpha`, `number of topics`, and `vocabulary size`

For example:
```
num_topics 20
num_terms 3126
alpha 0.015
```

Remember that `alpha` is the hyperparameter for document-topics learned from Mr. LDA. You can find its value from `20news.mrlda.train-lda/alpha-ITERATION` using the following command:

```
$ hadoop jar target/mrlda-0.9.0-SNAPSHOT-fatjar.jar \
   edu.umd.cloud9.io.ReadSequenceFile 20news.mrlda.train-lda/alpha-ITERATION
```

Finally, compute the held-out likelihood:

```
# Infer heldout for each documents
$ cd lda-c-dist
$ ./lda inf inf-settings.txt \
   ../20news.mrlda.train.20.ldac ../20news.ldac.test ../20news.mrlda.20.HL

# Average heldout scores
$ cd ..
$ python parse_20news/calculate_heldout_likelihood.py 20news.mrlda.20.HL-lda-lhood.dat
```

**Step 4: Compute Topic coherence**

We use [Topic Interpretability](https://github.com/jhlau/topic_interpretability) to compute topic coherence. Topic coherence evaluates topics against a ground corpus using measures such as `npmi` (stands for normalized point-wise mutual information). The ground corpus can be the whole `Wikipedia`. For this task, we use the training set and test set as a ground corpus.

Download the Topic Interpretability tool:

```
$ git clone https://github.com/jhlau/topic_interpretability
```

Prepare the corpus:

```
$ mkdir 20news_train_test_raws
$ cp -r 20news.raw.train 20news_train_test_raws/
$ cp -r 20news.raw.test 20news_train_test_raws/
```

Compute statistics:

```
# Ggenerate .oneline file
$ python parse_20news/createOneLineDict.py 20news.vocab.txt

# Get statistics of corpus
$ python topic_interpretability/ComputeWordCount.py 20news.vocab.txt.oneline \
    20news_train_test_raws > 20news.train.test.wc
```

If you want to use `Wikipedia` as a ground corpus, it is better than the directory that contains many small (1000 documents, one per line) files.

Prepare topics:

```
# Get topic file from HDFS
$ hadoop jar target/mrlda-0.9.0-SNAPSHOT-fatjar.jar cc.mrlda.DisplayTopic \
    -index 20news.mrlda.train-parsed/term \
    -input 20news.mrlda.train-lda/beta-ITERATION \
    -topdisplay 20 > 20news.mrlda.train.20.topics

# Convert it to proper format
$ python parse_20news/convertMrldaTopicsToTopics.py 20news.mrlda.train.20.topics \
    20news.mrlda.train.20.ti.topics 20
```

Compute Topic Coherence using `npmi`:

```
$ python topic_interpretability/ComputeObservedCoherence.py \
    20news.mrlda.train.20.ti.topics npmi 20news.train.test.wc > 20news.mrlda.20.oc
```

**WARNING**: The following documentation may be out of date...


Input Data Format
----------

The data format for Mr. LDA package is defined in class `Document.java` of every package. It consists an `HMapII.java` object, storing all word:count pairs in a document using an integer:integer hash map. **Take note that the word index starts from 1, whereas index 0 is reserved for system message.** Interesting user could refer following piece of code to convert an *indexed* document `String.java` to `Document.java`:

```java
String inputDocument = "Mr. LDA is a Latent Dirichlet Allocation topic modeling package based on Variational Bayesian learning approach using MapReduce and Hadoop";
Document outputDocument = new Document();
HMapII content = new HMapII();
StringTokenizer stk = new StringTokenizer(inputDocument);
while (stk.hasNext()) {
      content.increment(Integer.parseInt(stk.hasNext), 1);
}
outputDocument.setDocument(content);
```

By defalut, Mr. LDA accepts sequential file format only. The sequence file should be key-ed by a unique document ID of `IntWritable.java` type and value-d by the corresponding `Document.java` data type.

If you preprocessing the raw text using `ParseCorpus.java` command, the directory `/hadoop/index/document/output/directory/document` is the exact input to the following stage.

Latent Dirichlet Allocation
----------

The primary entry point of Mr. LDA package is via `VariationalInference.java` class. You may start training, resume training or launch testing on input data.

To print the help information and usage hints, please run the following command

    hadoop jar Mr.LDA.jar cc.mrlda.VariationalInference -help

To train LDA model on a dataset, please run one of the following command:

    hadoop jar Mr.LDA.jar cc.mrlda.VariationalInference -input /hadoop/index/document/output/directory/document -output /hadoop/mrlda/output/directory -term 60000 -topic 100
    hadoop jar Mr.LDA.jar cc.mrlda.VariationalInference -input /hadoop/index/document/output/directory/document -output /hadoop/mrlda/output/directory -term 60000 -topic 100 -iteration 40
    hadoop jar Mr.LDA.jar cc.mrlda.VariationalInference -input /hadoop/index/document/output/directory/document -output /hadoop/mrlda/output/directory -term 60000 -topic 100 -iteration 40 -mapper 50 -reducer 20
    hadoop jar Mr.LDA.jar cc.mrlda.VariationalInference -input /hadoop/index/document/output/directory/document -output /hadoop/mrlda/output/directory -term 60000 -topic 100 -iteration 40 -mapper 50 -reducer 20 -localmerge

The first four parameters are required options, and the following options are free parameter with their respective default values. Take note that `-term` option specifies the total number of unique tokens in the whole corpus. If this value is not available from context at run time, it is advised to set this option to the approximated upper bound of the total number of unique tokens in the entire corpus.

To resume training LDA model on a dataset, please run following command, it resumes Mr. LDA from iteration 5 to iteration 40:

    hadoop jar Mr.LDA.jar cc.mrlda.VariationalInference -input /hadoop/index/document/output/directory/document -output /hadoop/mrlda/output/directory -term 60000 -topic 100 -iteration 40 -modelindex 5

Take note that, to resume Mr. LDA learning, it requires the corresponding beta (distribution over tokens for a given topic), alpha (hyper-parameter for topic) and gamma (distribution over topics for a give document) to be presented.

To launch testing LDA model on a held-out dataset, please run the following command:

    hadoop jar Mr.LDA.jar cc.mrlda.VariationalInference -input /hadoop/index/document/output/directory/test-data -output /hadoop/mrlda/output/test-output -term 60000 -topic 100 -iteration 100 -modelindex 40 -test /hadoop/mrlda/output/directory

This command launches the testing of model after 40 iterations from the training output `/hadoop/mrlda/output/directory` and run 100 iteration on the testing data `/hadoop/index/document/output/directory/test-data`. Take note that `-test` option specifies the training output, and `-modelindex` specifies the model index from the training output.

Informed Prior
----------

Informed prior guild the latent Dirichlet allocation program to some topics which are particularly of interest. A typical informed prior word list looks like following, whereas *every* row is a set of words that belong (or "should" belong) to the same topic.
    
    foreign eastern western domestic immigration foreigners ethnic immigrants cultural culture easterns westerners westernstyle immigrant
    believe church hope believed determine christian religious christmas believes god determined fatal islamic faith christ jesus fate christopher christians churches belief religion gods christies fatalities saint islam beliefs faithful fatally determining bible lord ritual soul destined determination mosque churchs blessing destiny fatality christine saints godfather
    fighting fight battle challenge argued arguments fought challenger fighters threw dominated riot argument challenged fighter knife argue battles confrontation stones cruel challenges challenging battling disagreed disagree fights disagreement knives challengers domination battled dominate
    military war chief service army corp troops soldiers officer officers corps combat marine wars veterans soldier troop veteran marines
    private person identified personal concern concerned concerns basis natural affected affect identify nature identification tend character concerning identity personally affecting core characters naturalization characterized personality tendency selfdefense identities affects characteristics selfdetermination naturally foundations identical
    ...

Let us refer the above content as an informed prior file in HDFS --- `/hadoop/raw/text/input/informed-prior.txt`. To generate the Mr. LDA acceptalbe informed prior with the correct mapping of the word indexing, please run the following command

    hadoop jar Mr.LDA.jar cc.mrlda.InformedPrior -input /hadoop/raw/text/input/informed-prior.txt -output /hadoop/index/document/output/directory/prior -index /hadoop/index/document/output/directory/term
    
To print the help information and usage hits, please run the following command

    hadoop jar Mr.LDA.jar cc.mrlda.InformedPrior -help
    
By the end of the execution, you should get an informed prior file with correct index mapping, ready for training topics using Mr. LDA, for example,

    hadoop fs -ls /hadoop/index/document/output/directory/
    Found 4 items
    drwxr-xr-x   - user supergroup          0 2012-01-12 12:18 /hadoop/index/document/output/directory/document
    -rw-r--r--   3 user supergroup         57 2012-01-12 12:25 /hadoop/index/document/output/directory/prior
    -rw-r--r--   3 user supergroup        282 2012-01-12 12:18 /hadoop/index/document/output/directory/term
    -rw-r--r--   3 user supergroup        189 2012-01-12 12:18 /hadoop/index/document/output/directory/title

To train LDA model on a dataset with informed prior, please run the following command
    
    hadoop jar Mr.LDA.jar cc.mrlda.VariationalInference -input /hadoop/index/document/output/directory/document -informedprior /hadoop/index/document/output/directory/prior -output /hadoop/mrlda/output/directory -term 60000 -topic 100

After Running Experiments
----------

The output is a set of parameters in sequence file format. In your output folder, you will see a set of 'beta-\*' files, and 'alpha-\*' files, and 'document-\*' directory. 'alpha-\*' are the hyperparameters, 'beta-\*' are the distribution over words per topic and 'document-\*' are the topic distribution for each document, where ('*' is the iteration index).

To display the top 20 ranked words of each topic, access 'beta-*' file using following command

    hadoop jar Mr.LDA.jar cc.mrlda.DisplayTopic -input /hadoop/mrlda/output/directory/beta-* -term /hadoop/index/document/output/document/term -topdisplay 20

Please set the '-topdisplay' to an extremely large value to display all the words in each topic. Note that the output scores are sorted and in log scale.

To display the distribution over all topics for each document, access 'document-\*' file using following command

    hadoop jar Mr.LDA.jar cc.mrlda.DisplayDocument -input /path/to/document-*

To display the hyper-parameters, access alpha-\* file using following command

    hadoop jar Mr.LDA.jar edu.umd.cloud9.io.ReadSequenceFile /path/to/alpha-*

You may refer to -help options for further information.

