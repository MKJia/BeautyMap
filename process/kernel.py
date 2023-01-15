import string
import cupy as cp

def utils(resolution, width, height):
    util_preamble = string.Template(
    '''
    __device__ float transformFrame(float16 x, float16 y, float16 z, 
                                    float16 R0, float16 R1, float16 R2, float16 t)
    {
        return R0 * x + R1 * y + R2 * z + t;
    }

    __device__ float16 clamp(float16 x, float16 min_x, float16 max_x) 
    {
        return max(min(x, max_x), min_x);
    }

    __device__ int getIdxLine(float16 x, float16 center)
    {
        int i = round((x - center) / ${resolution});
        return i;
    }

    __device__ int getIdxLayer(float16 x, float16 y, float16 cx, float16 cy)
    {
        // Return 1D index of a point (x, y) in a layer
        int idx_x = getIdxLine(x, cx) + ${width} / 2;
        int idx_y = getIdxLine(y, cy) + ${height} / 2;

        // Check if the index is inside the map
        if (idx_x < 0 || idx_x >= ${width} || idx_y < 0 || idx_y >= ${height})
        {
            return -1;
        }
        return ${width} * idx_x + idx_y;
    }

    __device__ int getIdxBlock(int idx, int layer_n)
    {
        // Return 1D index of a point (x, y) in multi-layer map block
        return (int)${layer_size} * layer_n + idx;
    }
    '''
    ).substitute(resolution=resolution,
                 width=width, 
                 height=height,
                 layer_size=width*height)

    return util_preamble


def fillingKernel(resolution, width, height, min_sensor_distance_bk, min_sensor_distance_fn, max_sensor_distance, max_height_range):
    filling_kernel = cp.ElementwiseKernel(
        in_params='raw U points, U center_x, U center_y, raw U R, raw U t',
        out_params='raw U filling_mask',
        preamble=utils(resolution, width, height),
        operation=string.Template(
        '''
        // Read points
        U p_x = points[i * 3];
        U p_y = points[i * 3 + 1];
        U p_z = points[i * 3 + 2];

        // Sector filter
        U sector = atan2(p_y, p_x);
        U dist_sq = p_x*p_x + p_y*p_y;
        if ( (sector > 2.85 || sector < -2.85) && dist_sq < ${min_sensor_distance_bk_sq} ) { return; }
        if (dist_sq < ${min_sensor_distance_fn_sq}) { return; }
        if (dist_sq > ${max_sensor_distance_sq}) { return; }
        
        // Transform points from sensor frame to map frame 
        U m_x = transformFrame(p_x, p_y, p_z, R[0], R[1], R[2], t[0]);
        U m_y = transformFrame(p_x, p_y, p_z, R[3], R[4], R[5], t[1]);
        U m_z = transformFrame(p_x, p_y, p_z, R[6], R[7], R[8], t[2]);
        
        // Get height (z) in lidar frame
        U l_z = m_z - t[2];
        
        // Check if the point is in height range
        if (l_z > ${max_height_range}) { return; }

        // Check if the point is inside the local map (body centered)
        int idx = getIdxLayer(m_x, m_y, center_x, center_y);
        if (idx < 0) { return; }

        // Update filling state
        atomicExch(&filling_mask[getIdxBlock(idx, 0)], (float)1.0);   // Mark grid as filled
        '''
        ).substitute(
            min_sensor_distance_bk_sq=min_sensor_distance_bk**2, 
            min_sensor_distance_fn_sq=min_sensor_distance_fn**2, 
            max_sensor_distance_sq = max_sensor_distance**2,
            max_height_range=max_height_range),
            name='filling_kernel'
        )
                            
    return filling_kernel