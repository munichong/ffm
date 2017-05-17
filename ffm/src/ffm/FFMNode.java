package ffm;

/**
 * @author chenhuang
 *
 */
public class FFMNode { 
	/**
	 * Each value of a field, i.e., feature value, has one FFMNode.
	 */
	// field_num; cw87: field_index?
	public int f;
	// feature_num; cw87: feature_index?
	public int j;
	// value
	public float v;
	@Override
	public String toString() {
		return "FFMNode [f=" + f + ", j=" + j + ", v=" + v + "]";
	}
}
