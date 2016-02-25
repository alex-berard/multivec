import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;


public class Example {

	int label;
	SparseVector sv;
	
	public Example(int label, SparseVector sv) {
	    this.label = label;
	    this.sv = sv;
	}
	
	private static String readLineSkip(BufferedReader in) throws IOException {
	    String line = null;
	    while ((line = in.readLine())!= null) {
	        line = line.trim();
	        if (!line.equals("") && !line.startsWith("#")) {
	            break; 
	        }
	    }
	    return line;
	    
	}
	
	public static Example readExample(BufferedReader in) throws IOException {
	    String line = readLineSkip(in);
	    if (line == null) {
	        return null;
	    }
	    
	    line = line.trim();
	    String rec = line.split("\\s")[0];
	    
	    int label = Integer.parseInt(rec);
	    String vector = "";
	    if (line.length() > rec.length()) {
	        vector = line.substring(rec.length() + 1);
	    }
	    SparseVector sv = SparseVector.load(vector);
	    return new Example(label, sv);
	}
	
	public void saveExample(BufferedWriter out) throws IOException {
	    out.write(label + " " + sv.save() + System.getProperty("line.separator"));
	}

    @Override
    public String toString() {
        return "Example [label=" + label + ", sv=" + sv + "]";
    }
	
	
}
