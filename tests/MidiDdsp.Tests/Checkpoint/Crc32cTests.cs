using System.Text;
using MidiDdsp.Core.Checkpoint;

namespace MidiDdsp.Tests.Checkpoint;

public class Crc32cTests
{
    [Fact]
    public void MatchesStandardCheckValue()
    {
        // Standard CRC-32C check value for the ASCII string "123456789".
        Assert.Equal(0xE3069283u, TfCheckpoint.Crc32c(Encoding.ASCII.GetBytes("123456789")));
    }

    [Fact]
    public void EmptyInputIsZero()
    {
        Assert.Equal(0u, TfCheckpoint.Crc32c([]));
    }

    [Fact]
    public void HandlesLengthsThatAreNotMultiplesOfEight()
    {
        // Checked against a plain bit-by-bit reflected CRC-32C, independent of
        // the hardware (net8.0) and table-driven (net472) implementations.
        var data = Enumerable.Range(0, 37).Select(i => (byte)(i * 7)).ToArray();
        uint expected = 0xFFFFFFFF;
        foreach (var b in data)
        {
            expected ^= b;
            for (int bit = 0; bit < 8; bit++)
                expected = (expected & 1) != 0 ? (expected >> 1) ^ 0x82F63B78u : expected >> 1;
        }
        Assert.Equal(~expected, TfCheckpoint.Crc32c(data));
    }
}
