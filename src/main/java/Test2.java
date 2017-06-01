import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by goldfenrir on 16/04/2017.
 */
public class Test2 {
    public static void main(String[] args) {
        INDArray arr1 = Nd4j.zeros(1,5);
        String s=arr1.toString().replaceAll(" ","");
        s=s.substring(1,s.length()-1);
        String[] entries=s.split(",");

        for (int i=0;i<entries.length;i++){
            System.out.println(entries[i]);
        }


    }

}
