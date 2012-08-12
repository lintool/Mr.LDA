Mr. LDA
================
Mr. LDA is a Latent Dirichlet Allocation topic modeling package based on Variational Bayesian learning approach using MapReduce and Hadoop, developed by a [Cloud Computing Research Team] (http://lintool.github.com/Mr.LDA/docs/team.html) in [University of Maryland, College Park] (http://www.umd.edu).

Please download the latest version from our [GitHub repository](https://github.com/lintool/Mr.LDA).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------
Download the source code package (unzip if necessary) to directory `/home/directory/` in your own local file system (not HDFS). 
To download all the dependency packages, please run the following command

    cd /home/directory/Mr.LDA/
    ant

Jar all the .class files anddependency packages to `Mr.LDA.jar`. This can
either be accomplished manually or by running the following command

    cd /home/directory/Mr.LDA/
    ant export

The above command should create `bin/Mr.LDA-{version}.jar` with all of the
proper libraries.

Tokenizing and Indexing
----------

Mr. LDA takes raw text file as input, every row in the text file represents a stand-alone document. Document title and content are separated by a *tab* ('\t'), and words in the content are separated by a *space* (' '). The raw input text file should look like this:

    'Big Bang Theory' Brings Stephen Hawking on as Guest Star	'The Big Bang Theory' is getting a visit from Stephen Hawking. The renowned theoretical physicist will guest-star on the April 5 episode of the CBS comedy, the network said Monday. In the cameo, Hawking visits uber-geek Sheldon Cooper (Jim Parsons) at work 'to share his beautiful mind with his most ardent admirer,' according to CBS. Executive producer Bill Prady said that having Hawking on the show had long been a goal, though it seemed unattainable. When people would ask us who a dream guest star' for the show would be, we would always joke and say Stephen Hawking knowing that it was a long shot of astronomical proportions, Prady said. In fact, we're not exactly sure how we got him. It's the kind of mystery that could only be understood by, say, a Stephen Hawking. Hawking, known for his book A Brief History of Time, has appeared on television comedies before, albeit in voice work. Hawking has done a guest spot on 'Futurama' and appeared as himself on several episodes of 'The Simpsons.'
    The World's Best Gourmet Pizza: 'Tropical Pie' Wins Highest Honor	To make the world's best pizza you'll need dough, mozzarella cheese and some top shelf tequila. On Thursday, top pizza-makers from around the globe competed for the title of 'World's Best Pizza' at the International Pizza Expo in Las Vegas. At stake was $10,000 and the highest honor in the industry. This year's big winner was anything but traditional. The 'Tropical Pie' - a blend melted asiago and mozzarella cheese, topped with shrimps, thinly sliced and twisted limes, a fresh mango salsa, all resting on a rich pineapple cream sauce infused with Patron. The recipe, devised by mad pizza scientist Andrew Scudera of Goodfella's Brick Oven Pizza in Staten Island, was months in the making.ame up with idea to use tequila, but it was a collaboration,' Andrew tells Shine. 'Everyone here at the restaurant dived in and gave their input, helping to perfect the recipe by the time we brought it to the show.' The competition in Vegas was steep-particularly in the 'gourmet' category, where the Tropical Pie was entered.	
    ...

Mr. LDA relies on [Lucene] (http://lucene.apache.org/core/) to tokenize all the text. Please take note that the indexing process in Mr. LDA does *not* provide mechanism to filter out words based on their frequency. However, for more information, interested users could refer to the class `ParseCorpus.java`, which consists three steps. The filter could be introduced after the second step.

To tokenize, parse and index the raw text file, please run either the following command

    hadoop jar Mr.LDA.jar cc.mrlda.ParseCorpus -input /hadoop/raw/text/input/directory -output /hadoop/index/document/output/directory
    hadoop jar Mr.LDA.jar cc.mrlda.ParseCorpus -input /hadoop/raw/text/input/directory -output /hadoop/index/document/output/directory -mapper 10 -reducer 4

To print the help information and usage hints, please run the following command

    hadoop jar Mr.LDA.jar cc.mrlda.ParseCorpus -help

By the end of execution, you will end up with three files/dirtories in the specified output, for example,

    hadoop fs -ls /hadoop/index/document/output/directory/
    Found 3 items
    drwxr-xr-x   - user supergroup          0 2012-01-12 12:18 /hadoop/index/document/output/directory/document
    -rw-r--r--   3 user supergroup        282 2012-01-12 12:18 /hadoop/index/document/output/directory/term
    -rw-r--r--   3 user supergroup        189 2012-01-12 12:18 /hadoop/index/document/output/directory/title

File `/hadoop/index/document/output/directory/term` stores the mapping between a unique token and its unique integer ID. Similarly, `/hadoop/index/document/output/directory/title` stores the mapping between a document title to its unique integer ID. Both of these two files are in sequence file format, key-ed by `IntWritable.java` and value-d by `Text.java`. You may use the following command to browse a sequence file in general

     hadoop jar Mr.LDA.jar edu.umd.cloud9.io.ReadSequenceFile /hadoop/index/document/output/directory/term
     hadoop jar Mr.LDA.jar edu.umd.cloud9.io.ReadSequenceFile /hadoop/index/document/output/directory/term 20

and option '20' specifies the first 20 records to be displayed.

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
