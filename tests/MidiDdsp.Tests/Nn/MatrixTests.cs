using MidiDdsp.Core.Nn;

namespace MidiDdsp.Tests.Nn;

public class MatrixTests
{
    [Fact]
    public void NegativeRowsThrowsArgumentOutOfRangeException()
    {
        // The two-arg constructor forwards to the three-arg one via `new float[...]` in its own
        // constructor-initializer, which runs before either constructor body -- a negative row
        // count used to reach that allocation with a negative array length and fail with
        // OverflowException instead, before the intended validation ever ran.
        Assert.Throws<ArgumentOutOfRangeException>(() => new Matrix(-1, 2));
    }

    [Fact]
    public void NegativeColsThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Matrix(2, -1));
    }

    [Fact]
    public void ValidDimensionsConstructSuccessfully()
    {
        var matrix = new Matrix(2, 3);
        Assert.Equal(2, matrix.Rows);
        Assert.Equal(3, matrix.Cols);
        Assert.Equal(6, matrix.Data.Length);
    }
}
