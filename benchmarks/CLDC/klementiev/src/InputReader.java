import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;


public class InputReader {

	public InputReader() {
	    
	}
	List<Example> readData(String file) throws IOException {
	    BufferedReader in = new BufferedReader(new FileReader(file));
	    List<Example> data = new LinkedList<Example>();
	    Example ex;
	    while ((ex  = Example.readExample(in)) != null) {
	        data.add(ex);
	    }
	    return data;
	}
	
}
