import edu.stanford.nlp.process.WordTokenFactory;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.List;

import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

class TaggerDemo {

  private TaggerDemo() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 3) {
      System.err.println("usage: java TaggerDemo modelFile fileToTag outputPath");
      return;
    }
    BufferedReader reader = null;
    MaxentTagger tagger = new MaxentTagger(args[0]);
    reader = new BufferedReader(new FileReader(args[1]));
    //System.out.println("reader size");
    /////System.out.println(reader.size());
    List<List<HasWord>> sentences = new ArrayList<List<HasWord>> ();
    String line;
    int count = 0;
    while ( (line = reader.readLine() ) != null){
       //System.out.println(line);
       count += 1;
       List<HasWord> tokens = new ArrayList<>();
       PTBTokenizer<Word> tokenizer = new PTBTokenizer(new StringReader(line),new WordTokenFactory(),"");
       for (Word label;tokenizer.hasNext();)
       {
           tokens.add(tokenizer.next());
       }
       sentences.add(tokens);

    }

    //System.out.println("the size of sentences");
    //System.out.println(count);
    //System.out.println(sentences.size());

    List<String> posSentence = new ArrayList<String> ();
    BufferedWriter posWriter = new BufferedWriter(new FileWriter(args[2]));

    for (List<HasWord> sentence : sentences) {
      List<TaggedWord> tSentence = tagger.tagSentence(sentence);
      //System.out.println(Sentence.listToString(tSentence, false));
      posWriter.write(Sentence.listToString(tSentence, false));
      posWriter.write("\n");

    }
    posWriter.close();
    //System.out.println("now saving the pos tagging file to ....");
    //System.out.println(args[2]);

  }

}
