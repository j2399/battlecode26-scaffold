package econ5;

import battlecode.common.MapLocation;

public class FastSet {
    private StringBuilder values;

    public FastSet() {
        values = new StringBuilder();
    }

    public boolean add(char value) {
        String str = String.valueOf(value);
        if (values.indexOf(str) == -1) {
            values.append(str);
            return true;
        }

        return false;
    }

    public boolean contains(char value) {
        return values.indexOf(String.valueOf(value)) > -1;
    }

    public boolean add(MapLocation location) {
        return add(encodeLocation(location));
    }

    public boolean remove(MapLocation location) {
        char encodedLocation = encodeLocation(location);
        int index = values.indexOf(String.valueOf(encodedLocation));
        if (index != -1) {
            values.deleteCharAt(index);
            return true;
        }
        return false;
    }

    public boolean contains(MapLocation location) {
        return contains(encodeLocation(location));
    }

    private char encodeLocation(MapLocation location) {
        return (char) ((location.x << 6) | location.y);
    }

    public char[] getValues() {
        return values.toString().toCharArray();
    }

    // Method to get all MapLocations
    public MapLocation[] getLocations() {
        char[] chars = getValues();
        MapLocation[] locations = new MapLocation[chars.length];
        for (int i = 0; i < chars.length; i++) {
            locations[i] = decodeLocation(chars[i]);
        }
        return locations;
    }

    private MapLocation decodeLocation(char encoded) {
        return new MapLocation(encoded >> 6, encoded & 0x3F);
    }
}