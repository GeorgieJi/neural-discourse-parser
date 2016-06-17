
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
    if (args.length != 2) {
      System.err.println("usage: java TaggerDemo modelFile fileToTag");
      return;
    }
    BufferedReader reader = null;
    MaxentTagger tagger = new MaxentTagger(args[0]);
    reader = new BufferedReader(new FileReader(args[1]));
    System.out.println("reader size");
    /////System.out.println(reader.size());
    String line;
    int count = 0;
    List<List<HasWord>> sentences = null;
    while ( (line = reader.readLine() ) != null){
       //System.out.println(line);
       count += 1;
       sentences.add(MaxentTagger.tokenizeText(line));

    }

    System.out.println("the size of sentences");
    System.out.println(sentences.size());
    for (List<HasWord> sentence : sentences) {
      List<TaggedWord> tSentence = tagger.tagSentence(sentence);
      System.out.println(Sentence.listToString(tSentence, false));
    }
  }

}
