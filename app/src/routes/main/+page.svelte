<script>
    import { goto } from '$app/navigation';
    let room;
</script>

<div class="h-screen w-screen flex flex-col justify-center items-center bg-gray-50">
    <div class="w-full max-w-lg">
        <input 
            type="text" 
            bind:value={room} 
            placeholder="Enter room name" 
            class="w-full p-4 text-lg border-b-2 border-gray-300 focus:outline-none focus:border-gray-400 bg-transparent mb-8"
        />

        <div 
            on:click={() => goto(`/play?room=${room}`)} 
            class="cursor-pointer w-full py-4 bg-gray-200 text-center text-xl text-gray-600 rounded-lg mb-4 hover:bg-gray-300 transition">
            Join Room
        </div>

        <div 
            on:click={() => goto(`/host?room=${room}`)} 
            class="cursor-pointer w-full py-4 bg-gray-200 text-center text-xl text-gray-600 rounded-lg hover:bg-gray-300 transition">
            Host Room
        </div>
    </div>
</div>
