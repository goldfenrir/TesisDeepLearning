
import edu.stanford.nlp.naturalli.NaturalLogicAnnotator;
import edu.stanford.nlp.naturalli.OpenIE;
import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.naturalli.NaturalLogicAnnotations;
import edu.stanford.nlp.naturalli.SentenceFragment;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;


import java.util.*;


/**
 *
 * @author  Jaime Diego Bustamante Arce
 * @date    15/04/2017
 */
public class MainEstructuracion {
    protected static StanfordCoreNLP pipeline = new StanfordCoreNLP(new Properties() {{
        setProperty("annotators", "tokenize,ssplit,pos,lemma,depparse,natlog,openie");
        setProperty("openie.splitter.threshold", "0.25");
        setProperty("openie.ignoreaffinity", "false");
        setProperty("openie.max_entailments_per_clause", "1000");
        setProperty("openie.triple.strict", "true");
        setProperty("ssplit.isOneSentence", "true");
        setProperty("tokenize.class", "PTBTokenizer");
        setProperty("tokenize.language", "en");
        setProperty("enforceRequirements", "true");
    }});

    public CoreMap annotate(String text) {
        Annotation ann = new Annotation(text);
        pipeline.annotate(ann);
        return ann.get(CoreAnnotations.SentencesAnnotation.class).get(0);
    }

    public boolean extraerEstructura(ArrayList<String> estructuras, String texto) {
        boolean found = false;
        if (estructuras==null) estructuras=new ArrayList<String>();
        Collection<RelationTriple> extractions = annotate(texto).get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
        for (RelationTriple extraction : extractions) {
            String s=new String();
            s=extraction.toString().replace("\t", ";");
            s=s.substring(s.indexOf(";")+1);
            s=s.trim();
            estructuras.add(s);
            System.out.println(s);
            found=true;
        }
        return found;
    }



    public void assertExtracted(String expected, String text) {
        boolean found = false;
        Collection<RelationTriple> extractions = annotate(text).get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
        for (RelationTriple extraction : extractions) {
            System.out.println(extraction.toString().replace("\t", "; "));
            if (extraction.toString().equals("1.0\t" + expected)) {
                System.out.println(extraction.toString());
                found = true;
            }
        }
        // assertTrue("The extraction (" + expected.replace("\t", "; ") + ") was not found in '" + text + "'", found);
    }

    public void assertExtracted(Set<String> expectedSet, String text) {
        Collection<RelationTriple> extractions = annotate(text).get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
        String actual = StringUtils.join(
            extractions.stream().map(x -> x.toString().substring(x.toString().indexOf("\t") + 1).toLowerCase()).sorted(),
            "\n");
        String expected = StringUtils.join(expectedSet.stream().map(String::toLowerCase).sorted(), "\n");
        // assertEquals(expected, actual);
    }

    public void assertEntailed(String expected, String text) {
        boolean found = false;
        Collection<SentenceFragment> extractions = annotate(text).get(NaturalLogicAnnotations.EntailedSentencesAnnotation.class);
        for (SentenceFragment extraction : extractions) {
            if (extraction.toString().equals(expected)) {
                found = true;
            }
        }
        //  assertTrue("The sentence '" + expected + "' was not entailed from '" + text + "'", found);
    }

}
