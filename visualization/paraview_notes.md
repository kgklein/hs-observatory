# ParaView Notes

The exporter writes `reconstruction_structured.vts` for field rendering plus
legacy ASCII VTK PolyData files for point-cloud and trajectory helpers.

Suggested display setup:

1. Open `reconstruction_structured.vts`.
2. Try ParaView's Slice, Clip, Threshold, Contour, or Volume Rendering filters.
3. Color by `Bmag_reconstructed`.
4. Open `spacecraft_tracks.vtk` and set a thicker line width.
5. Open `spacecraft_points.vtk` and color by `spacecraft_index`.
6. Open `barycenter_path.vtk` and `reconstruction_cylinder.vtk`.
7. If validation arrays exist, switch the field color array to
   `error_magnitude` or `error_percent`.

The `cylinder_mask` array marks points inside the reconstruction cylinder. Use a
Threshold filter on `cylinder_mask > 0.5` if you want to hide the rectangular
corners of the structured local grid.
