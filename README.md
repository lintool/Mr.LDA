Mr. LDA
================
Mr. LDA is a Latent Dirichlet Allocation topic modeling package based on Variational Bayesian learning approach using MapReduce and Hadoop, developed by a [Cloud Computing Research Team] (http://lintool.github.com/Mr.LDA/docs/team.html) in [University of Maryland, College Park] (http://www.umd.edu).

Please download the latest version from our [GitHub repository](https://github.com/lintool/Mr.LDA).

Tokenizing and Indexing
----------

Mr. LDA takes raw text file as input, every row in the text file represents a stand-alone document. Document title and content are separated by a *tab* ('\t'), and words in the content are separated by a *space* (' '). The raw input text file should look like this:

	'Big Bang Theory' Brings Stephen Hawking on as Guest Star	'The Big Bang Theory' is getting a visit from Stephen Hawking. The renowned theoretical physicist will guest-star on the April 5 episode of the CBS comedy, the network said Monday. In the cameo, Hawking visits uber-geek Sheldon Cooper (Jim Parsons) at work 'to share his beautiful mind with his most ardent admirer,' according to CBS. Executive producer Bill Prady said that having Hawking on the show had long been a goal, though it seemed unattainable. When people would ask us who a Ôdream guest star' for the show would be, we would always joke and say Stephen Hawking Ð knowing that it was a long shot of astronomical proportions,Ó Prady said. ÒIn fact, we're not exactly sure how we got him. It's the kind of mystery that could only be understood by, say, a Stephen Hawking.Ó Hawking, known for his book A Brief History of Time, has appeared on television comedies before, albeit in voice work. Hawking has done a guest spot on 'Futurama' and appeared as himself on several episodes of 'The Simpsons.'
	The World's Best Gourmet Pizza: 'Tropical Pie' Wins Highest Honor	To make the world's best pizza you'll need dough, mozzarella cheese and some top shelf tequila. On Thursday, top pizza-makers from around the globe competed for the title of 'World's Best Pizza' at the International Pizza Expo in Las Vegas. At stake was $10,000 and the highest honor in the industry. This year's big winner was anything but traditional. The 'Tropical Pie' - a blend melted asiago and mozzarella cheese, topped with shrimps, thinly sliced and twisted limes, a fresh mango salsa, all resting on a rich pineapple cream sauce infused with Patron. The recipe, devised by mad pizza scientist Andrew Scudera of Goodfella's Brick Oven Pizza in Staten Island, was months in the making.ame up with idea to use tequila, but it was a collaboration,' Andrew tells Shine. 'Everyone here at the restaurant dived in and gave their input, helping to perfect the recipe by the time we brought it to the show.' The competition in Vegas was steep-particularly in the 'gourmet' category, where the Tropical Pie was entered.	
	...

Mr. LDA relies on [Lucene] (http://lucene.apache.org/core/) to tokenize all the text. Please take note that the indexing process in Mr. LDA does *not* provide mechanism to filter out words based on their frequency. However, for more information, interested users could refer to the class `ParseCorpus.java`, which consists three steps. The filter could be introduced after the second step.

To tokenize, parse and index the raw text file, please run the following command

    hadoop jar Mr.LDA.jar cc.mrlda.ParseCorpus -input /hadoop/raw/text/input/directory -output /hadoop/raw/text/output/directory
    hadoop jar Mr.LDA.jar cc.mrlda.ParseCorpus -input /hadoop/raw/text/input/directory -output /hadoop/raw/text/output/directory -mapper 10 -reducer 4

To print the help information and usage hints, please run the following command

    hadoop jar Mr.LDA.jar cc.mrlda.ParseCorpus -help

Input Data Format
----------

The data format for Mr. LDA package is defined in class `Document.java` of every package. It consists an `HMapII` object, storing all word:count pairs in a document using an integer:integer hash map. **Take note that the word index starts from 1, whereas index 0 is reserved for system message.** Interesting user could refer following piece of code to convert an *indexed* document `String` to `Document`:

```java
	String inputDocument = "1 2 1 8 1 9 8 4 1 1 2 1 9 8 6";
	Document outputDocument = new Document();
	HMapII content = new HMapII();
	StringTokenizer stk = new StringTokenizer(inputDocument);
	while (stk.hasNext()) {
	      content.increment(Integer.parseInt(stk.hasNext), 1);
	}
	outputDocument.setDocument(content);
```

By defalut, Mr. LDA accepts sequential file format only. The sequence file should be key-ed by a unique document ID of `IntWritable.java` type and value-d by the corresponding `Document.java` data type.