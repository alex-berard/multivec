


public class SparseVector {
	int ids[];
	double values[];
	
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < ids.length; i++) {
			sb.append(ids[i]);
			sb.append(":");
			sb.append(values[i]);
			sb.append(" ");
		}
		
		String s = sb.toString().trim();
		return s;
	}

	public String save() {
		return toString();
	}
	
	private SparseVector() {
		
	}
	
	public SparseVector(int initUpTo) {
		ids = new int[initUpTo + 1];
		values = new double[initUpTo + 1];
		for (int  i = 0; i < ids.length; i++) {
			ids[i] = i;
		}
	}
	
	public static SparseVector load(String line) {
		line = line.trim();
		SparseVector s = new SparseVector();
		if (line.equals("")) {
			s.ids = new int[0];
			s.values = new double[0];
		}
			
		String fields[] = line.split("\\s+");
		s.ids = new int[fields.length];
		//s.ids = new double[fields.length];
		s.values = new double[fields.length];
		for (int i = 0; i < fields.length; i++) {
			String pair[] = fields[i].split(":");
			s.ids[i] = Integer.parseInt(pair[0]);
			//s.ids[i] = Double.parseDouble(pair[0]);
			s.values[i] = Double.parseDouble(pair[1]);
		}
		return s;
	}
	
	public int topFeatureId() {
		return ids[ids.length - 1];
	}
	
	
	public void scale(double coeff)  {
		
		for (int i = 0; i < ids.length; i++) {
			values[i] *= coeff;
		}
	}

	// add to non sparse vector
	public void add(SparseVector v, double coeff) {
		for (int i = 0; i < v.ids.length; i++) {
			int id = v.ids[i];
			
			// check that current vector is not a sparse vector
			assert (id < ids.length && ids[id] == id);
			
			values[id] += v.values[i] * coeff; 
		}
	}
	
	
	// this vector is non sparse
	public double innerProductForNonSparse(SparseVector sparseV) {
		double sum = 0;
		
		for (int i = 0; i < sparseV.ids.length; i++) {
			int id = sparseV.ids[i];
			
			if (id >= ids.length) {
			    continue;
			}
			// check that current vector is not a sparse vector
			assert (ids[id] == id);
			
			sum +=  values[id] * sparseV.values[i]; 
		}

		return sum;
		
	}
	
	public static double innerProduct(SparseVector a, SparseVector b) {
		
		double sum = 0;
		int iA = 0, iB = 0;
		
		while (true) {
			if (iA >= a.ids.length || iB >= b.ids.length) {
				break;
			}
			
			if (a.ids[iA] < b.ids[iB]) {
				iA++;
			} else if (a.ids[iA] > b.ids[iB]) {
				iB++;
			} else {
				sum += a.values[iA++] * b.values[iB++];
			}
		}
		return sum;
	}

	
	public static void main(String args[]) {
		SparseVector a = SparseVector.load("1:0.13232  3:12.e-5  6:10");
		SparseVector b = SparseVector.load("3:1 7:2 16:11");
		
		System.out.println(SparseVector.innerProduct(a, b));
		
		SparseVector c = new SparseVector(b.topFeatureId());
		c.add(a, 1.);
		c.add(b, -1.);
		c.scale(2);
		System.out.println(c);
	}
	
}
